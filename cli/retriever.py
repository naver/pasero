#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import json
import torch
import os
import sys
from flask import Flask, abort, jsonify, request
from waitress import serve
from unidecode import unidecode
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import string


app = Flask(__name__)

description = r"""Index collections and make them searchable through HTTP, for use with `pasero-serve`.
Note that this is a very basic and inefficient retrieval-augmented generation demo: the retriever only supports 
BM25 for now, and the index also needs to fit in memory. As such, this is not usable with large collections like 
wikipedia. For more efficient retrieval, consider using Pyserini.

Usage:
```bash
# Install required packages
pip install rank_bm25 nltk unidecode
# Start retriever in background
pasero-retriever data/collection.jsonl --port 8001 &
# Serve retrieval-augmented LLM
pasero-serve models/rag_model --retriever-url http://localhost:8001 --port 8001
```

The JSONL collections should contain one JSON dict per line, corresponding to a document, with at least 'title' and 
'text' fields, and optionally 'rich_text', 'timestamp' and 'url' fields.
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=description)
parser.add_argument('collections', nargs='+', help='paths to JSONL collections to index with BM25')
parser.add_argument('--port', type=int, default=8001, help='listen for HTTP on this port')
parser.add_argument('--save-index', type=bool, action=argparse.BooleanOptionalAction, help='whether to save/load the '
                    "BM25 indexes with pickle (as 'COLLECTION.pickle')")


@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400


class BM25:
    def __init__(self, documents: list[dict]):
        self._index = None
        self._documents = documents

    @staticmethod
    def preprocess(content: str) -> list[str]:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(content)
        tokens = [unidecode(token.lower()) for token in tokens]  # normalize by removing accents and lowercasing
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]  # remove
        # stop words and punctuation
        return tokens

    def index(self, template: str = '{title} {text}'):
        indexed_contents = []
        for doc in self._documents:
            contents = template.format(**doc)
            contents = self.preprocess(contents)
            indexed_contents.append(contents)
        self._index = BM25Okapi(indexed_contents)

    def save_index(self, path):
        torch.save(self._index, path)
    def load_index(self, path):
        self._index = torch.load(path)

    def retrieve(self, query: str, topk: int) -> list[dict]:
        query_tokens = self.preprocess(query)
        scores = self._index.get_scores(query_tokens)
        ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        return [self._documents[i] for i in ids]


@app.route('/get_search_models', methods=['GET', 'POST'])
def get_search_models():
    return jsonify([info for info, _ in retrievers])


@app.route('/search/<collection>', methods=['GET', 'POST'])
def search(collection: str):
    data = request.json
    model = data['model']
    topk = data['nbdoc']
    query = data['query']

    for info, retriever in retrievers:
        if info['collection'] == collection and info['model'] == model:
            results = retriever.retrieve(query, topk)
            return jsonify(results)
    abort(400, f"unknown collection ('{collection}') or model ('{model}')")


def main():
    global retrievers
    retrievers = []

    args = parser.parse_args()
    nltk.download('stopwords')

    for path in args.collections:
        collection = os.path.basename(path).removesuffix('.jsonl')  # 'data/collection.jsonl' -> 'collection'
        print(f"loading documents from '{path}'", file=sys.stderr)
        documents = [json.loads(line) for line in open(path)]
        assert all('text' in doc and 'title' in doc for doc in documents), "missing 'text' or 'title' field"

        retriever = BM25(documents)  # only supported model for now
        index_path = path.removesuffix('.jsonl') + '.pickle'
        
        if args.save_index and os.path.exists(index_path):
            print(f"loading index from '{index_path}'", file=sys.stderr)
            retriever.load_index(index_path)
        else:
            print(f"indexing '{path}'", file=sys.stderr)
            retriever.index(template='{title} {title} {text}')  # give more weight to the title
            if args.save_index:
                print(f"saving index as '{index_path}'")
                retriever.save_index(index_path)
        
        retrievers.append((
            {'collection': collection, 'model': 'bm25', 'description': f'{collection} (BM25)'},
            retriever,
        ))

    print(f"serving on http://localhost:{args.port}", file=sys.stderr)
    serve(app, host="0.0.0.0", port=args.port)

if __name__ == '__main__':
    main()
