#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import sys
import logging
import argparse
import time
import itertools
import functools
import json
import traceback
import regex
import requests
from datetime import datetime
from numbers import Number
from torch import multiprocessing as mp
from flask import Flask, abort, jsonify, request, render_template, Response
from typing import Optional
from waitress import serve
from copy import deepcopy

app = Flask(__name__, template_folder='.')

from pasero import utils
from pasero.decoding import TextGenerator


parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+', help='paths of the models to serve (model directory or checkpoint). Each '
                    'model can be followed by a device id following this format: "PATH:GPU_ID"', default=[])
parser.add_argument('--port', type=int, default=8000, help='listen for HTTP on this port')
parser.add_argument('--dtype', help="override the models' default type with this one")
parser.add_argument('--retriever-url', help='address of a retrieval server started with `pasero-retriever`')

utils.init_logging(stream=sys.stderr)
logger = logging.getLogger('server')


actions = {
    '/help': 'get a list of available actions and options with their description',
    '/list': 'get a list of all available models',
    '/retrievers': 'get a list of all available retrievers',
    '<model>/decode': 'do batched decoding given inputs with given model',
    '<model>/stream': 'stream tokens with given model from given input',
    '<model>/info': 'get information about given model (task, covered languages, etc.)',
}

options = {
    'input': (list, 'text input to translate or use as prompt'),
    'beam_size': (int, 'beam size in beam search decoding (default: 5)'),
    'sampling': (bool, 'use sampling instead of greedy decoding'),
    'sampling_topp': (float, 'do nucleus sampling with this top-p value'),
    'sampling_topk': (int, 'do top-k sampling with this k value'),
    'sampling_temperature': (float, 'do sampling with this softmax temperature'),
    'repeat_penalty': (float, 'penalty for repeated tokens (default: 1)'),
    'max_output_len': (int, 'maximum number of tokens to generate'),
    'source_lang': (str, 'source language'),
    'target_lang': (str, 'target language'),
    'stop_regex': (str, 'regular-expression that will stop generation when matched'),
    'retriever_id': (str, 'look for relevant documents with this retriever and append them to the prompt'),
    'retriever_output_template': (str, 'what template to use to format the user query and retrieved documents (should contain {user_msg} and {docs} placeholders)'),
    'retriever_query_template': (str, 'prompt the LLM to generate a query with this template (should contain {user_msg} and {retrieve} placeholders)'),
    'retriever_auto_template': (str, 'ask the LLM to decide whether to retrieve with this template (should contain a {user_msg} placeholder), leave empty to always retrieve'),
    'retriever_topk': (int, 'how many documents to retrieve'),
}

retriever_url = None
models = {}


def process_input(input: list[str]) -> list[str]:
    if not input:
        return []
    else:
        return [line.replace('\r', '') for line in input if line.strip()]



@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400



def get_model(name):
    model = models.get(name)
    if model is None:
        abort(400, f'model {name} does not exist')
    return model


def strtobool(val: str) -> bool:
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        abort(400, f"invalid boolean value '{val}'")


def get_data() -> dict:
    """ Read and parse HTTP request parameters """
    if request.method == 'GET':
        data = request.args
    else:
        data = request.form

    input = data.getlist('input[]') or data.getlist('input')
    data = data.to_dict()
    data.pop('input[]', None)
    data.pop('input', None)
    if input:
        data['input'] = input  # list[str]

    # convert all the options to the right type
    clean_data = {}
    for name, value in data.items():
        if name in options:
            type, _ = options[name]
            if type in (int, float, list, str):
                value = type(value)
            elif type == bool:
                value = strtobool(value)
            else:
                abort(400, f"unknown type '{type}'")
            clean_data[name] = value
        else:
            abort(400, f"unknown option '{name}'")
    data = clean_data

    retriever_config = {}
    # retriever options are those starting with "retriever_": remove them from `data` and add them to `retriever_config`
    for key in list(data):
        if key.startswith('retriever_'):
            value = data.pop(key)
            if isinstance(value, str):
                value = value.replace('\\n', '\n')
            if value is not None:
                retriever_config[key.removeprefix('retriever_')] = value

    data['max_output_len'] = max(1, data.get('max_output_len', 100))

    if 'id' in retriever_config:  # retrieval is activated if a retriever id is given
        if retriever_config['id'] == 'none':
            retriever_config['id'] = None
        data['retriever_config'] = retriever_config
    
    return data


def parse_device(device: str) -> str:
    # {'cpu', 'cuda', 'cuda:X', 'X'} -> {'cpu', 'cuda:X'}
    if device == 'cpu':
        return device
    elif device == 'cuda':
        return 'cuda:0'
    device = int(device.removeprefix('cuda:'))
    return f'cuda:{device}'


def parse_devices(devices: str) -> list[str]:
    # '0,1' -> ['cuda:0', 'cuda:1']
    if not devices:
        return ['cuda:0']
    return [parse_device(device) for device in devices.split(',')]


@app.route('/<name>', methods=['GET', 'POST'])
def playground(name: str):
    model = get_model(name)  # error if model does not exist
    # `name` is not used here, but parsed by javascript in 'playground.html' to query information about the model
    # with "/<name>/model_info"
    return render_template('playground.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template(
        'index.html',
        models=[model.model_info for model in models.values()],
    )


def set_defaults(model_info: dict) -> dict:
    """
    Update the decoding options to set a source language and target language using the default languages of the model
    if any, or English to French otherwise.
    """
    model_info = deepcopy(model_info)  # avoid modifying the model info dict directly, which is model property and can 
    # be accessed concurrently
    decoding_opts = model_info.setdefault('decoding_options', {})

    default_source_lang = model_info.get('default_source_lang')
    source_langs = model_info.get('source_langs') or []
    for lang in default_source_lang, 'fr', 'fra_Latn':
        if lang and lang in source_langs:
            decoding_opts['source_lang'] = lang
            break
 
    default_target_lang = model_info.get('default_target_lang')
    target_langs = model_info.get('target_langs') or []
    for lang in default_target_lang, 'en', 'eng_Latn':
        if lang and lang in target_langs:
            decoding_opts['target_lang'] = lang
            break
    
    model_info['retriever_config'] = get_retriever_config(model_info)
    return model_info


@app.route('/<name>/api', methods=['GET', 'POST'])
def api(name: str):
    model = get_model(name)
    model_info = set_defaults(model.model_info)  # copies the dict
    # For pretty printing:
    model_info['decoding_options'] = {repr(k): repr(v) for k, v in model_info['decoding_options'].items()}
    
    return render_template(
        'api.html',
        model_info=model_info,
    )


def get_retriever_config(model_info: dict) -> dict:
    """
    Create a retriever configuration for `prompt_with_retrieval()` using the model's configuration (as specified in
    its `inference.yaml` file) or the default values specified here.
    
    This configuration will be used as default in the playground.
    """
    retriever_config = {
        'topk': 1,  # how many documents to retrieve
        'id': 'none',  # id of the retriever in the format "collection/model" (e.g., "wikipedia/bm25")
        'min_doc_len': 1,  # documents with fewer tokens than this won't be returned
        'max_doc_len': 4096,  # max total number of tokens for the retrieved documents (should be smaller than the 
        # model's context size)
        # Template to prompt the LLM with the retrieved documents to answer the user's question (the conversation 
        # history is also prepended):
        'output_template': (
            '{user_msg}\nYou can use the following documents to guide your answer:\n\n'
            '{docs}\n\n'
            'Given the above documents, please answer the following query as faithfully as possible: {user_msg}'
        ),
        # Template for prompting the LLM to generate a query for the retriever ('{retrieve}' is replaced by a 
        # hard-coded format):
        'query_template': (
            "Please generate a search engine query for retrieving documents that are relevant to the following "
            "question, using the format {retrieve}: {user_msg}"
        ),
    }  # default values, can be overriden by the model's config

    # An "auto_template" can also be provided (containing a "{user_msg}" placeholder). For instance:
    # 'Do you think that having access to up-to-date Wikipedia pages would help you answer the following question 
    # accurately? "{user_msg}"'
    
    retriever_config.update(
        model_info.get('retriever_config') or {},
    )  # override the default values above by the model's configuration, if any
    if retriever_url:
        retriever_config['url'] = retriever_url  # overrides the model's config
    return retriever_config


@app.route('/<name>/info', methods=['GET', 'POST'])
def model_info(name: str):
    if name not in models:
        return Response(status=204)
    model = models[name]
    return jsonify(set_defaults(model.model_info))


@app.route('/list', methods=['GET', 'POST'])
def list_models():
    return jsonify([model.model_info for model in models.values()])


@app.route('/<name>/retrievers', methods=['GET', 'POST'])
def list_retrievers(name: str):
    retriever_config = get_retriever_config(models[name].model_info)
    url = retriever_config.get('url')
    if url is None:
        retrievers = []
    else:
        retrievers = requests.post(f'{url}/get_search_models').json()
    return jsonify(retrievers)


@app.route('/help', methods=['GET', 'POST'])
def help():
    return jsonify({'actions': actions, 'options': {name: desc for name, (_, desc) in options.items()}})


def clean_document(content: str) -> str:
    """
    Clean retrieved documents by merging consecutive line breaks into a single line break. This makes delimitation
    between documents clearer.
    """
    content = regex.sub(r'\s*\n+', '\n', content, flags=regex.DOTALL)
    content = content.strip()
    return content


def retrieve(query: str, retriever_config: dict) -> list[dict[str, str]]:
    """
    Retrieve documents matching given query with given configuration.
    """
    try:
        retriever_id = retriever_config['id']
        url = retriever_config['url']  # location of the retrieval server
        topk = retriever_config.get('topk') or 1  # how many documents to retrieve
        collection, model = retriever_id.strip().rsplit('/', maxsplit=1)  # e.g., "wikipedia/bm25"

        results = requests.post(
            f'{url}/search/{collection}',
            json={
                'query': query,
                'nbdoc': topk * 2,  # retrieve more documents because some may be too short or too long (and will be 
                # skipped in `prompt_with_retrieval()`)
                'model': model,
            },
        ).json()
        
        if isinstance(results, dict):
            results = results['result']

        docs = []
        for doc in results:
            
            title = (doc.get('title') or '').strip()
            content = clean_document(
                doc.get('rich_text') or doc.get('markdown') or doc.get('text')  # use the first non-empty content field
            )
            url = (doc.get('url') or '').strip()
            if 'timestamp' in doc:  # convert timestamp to a readable date
                date = datetime.fromtimestamp(doc['timestamp'])
                date = date.strftime('%B %-d %Y')
            else:
                date = None
            
            if not content:  # skip empty documents
                continue

            # Concatenate the non-empty fields into a single string that will be used to prompt the LLM:
            formatted = []
            if title:
                formatted.append(f'## Title: {title}')
            if url:
                formatted.append(f'## URL: {url}')
            if date:
                formatted.append(f'## Date: {date}')
            formatted.append(content)
            formatted = '\n'.join(formatted)
            doc = {
                'formatted': formatted,  # this field will be used to prompt the LLM, the other fields will be used
                # to show the retrieved documents to the user
                'url': url,
                'content': content,
                'title': title,
            }
            docs.append(doc)
        return docs
    except:  # if retrieval fails, fall back to non-retrieval-augmented generation
        traceback.print_exc()
        return []


def generate_query(
    model: TextGenerator,
    history: str,   # previous turns of the conversation (excluding the last user message), ends with "User:"
    user_msg: str,  # last user message without the "User:" prompt
    config: dict,   # decoding configuration
    retriever_config: Optional[dict] = None,
) -> Optional[str]:
    r"""
    Use the given model to decide whether it needs retrieval to answer the user's message. If yes, use the user's 
    message and conversation history to generate a query for the retriever. Return the query, or `None` if retrieval 
    should be skipped.
    """
    if not user_msg or not retriever_config or retriever_config['id'] is None:  # skip if retrieval is disabled
        return None

    auto_template = retriever_config.get('auto_template')
    yes = retriever_config.get('yes') or 'Yes'  # custom yes or no answers for the auto-retrieve feature
    no = retriever_config.get('no') or 'No'

    if auto_template:  # let the LLM decide whether it should retrieve or not
        if '{user_msg}' not in auto_template:  # add the {user_msg} placeholder if missing
            auto_template = auto_template + ' {user_msg}'
        prompt = history + auto_template.format(user_msg=user_msg)
        yes = f'{prompt}\nAssistant: {yes}'
        no = f'{prompt}\nAssistant: {no}'
        # Example prompt:
        """
        User: Who is the prime minister of France?
        Assistant: Gabriel Attal
        User: Do you you think that having access to up-to-date Wikipedia pages would help you answer the following 
        question accurately? "How old is he?"
        Assistant: Yes
        """  # vs "[...] Assistant: No"
        logger.info(f'Deciding whether to retrieve with the prompt: {repr(yes)}')
        out = model.decode(yes, no, max_output_len=0, return_scores=True)
        yes_score = out[0][0]['score']
        no_score = out[1][0]['score']
        if yes_score > no_score:  # the lower the better
            logger.info(f'The LLM decided not to retrieve: {yes_score:.2f} > {no_score:.2f}')
            return None
        else:
            logger.info(f'The LLM decided to retrieve: {yes_score:.2f} < {no_score:.2f}')
    
    query_template = retriever_config.get('query_template')
    if not query_template:  # if there is no query generation template, the last user message is used as query for 
        # the retriever. In the example above, this would be "How old is he?"
        return user_msg
    if '{user_msg}' not in query_template or '{retrieve}' not in query_template:
        logger.warning('The query template is not in a valid format')

    prompt = history + query_template.format(
        user_msg=user_msg,
        retrieve='RETRIEVE("Your query")'
    ) + '\nAssistant: RETRIEVE("'  # prompt the assistant with "RETRIEVE(" to force it to respect the format

    # Example prompt:
    """
    User: Who is the prime minister of France?
    Assistant: Gabriel Attal
    User: Please generate a search engine query for retrieving documents that are relevant to the following 
    question, using the format RETRIEVE("Your query"): How old is he?
    Assistant: RETRIEVE("
    """
    logger.info(f'Attempting to generate a query with prompt: {repr(prompt)}')
    pattern = regex.compile(r'(?P<query>.+?)"')  # the LLM's answer should match this regex
    result = ''

    # Greedy search for max 100 tokens:
    # config = {'max_output_len': 100, 'sampling': False, 'beam_size': 1}
    
    for out in model.stream(prompt, **config):
        result += out['detok']
        if (m := regex.search(pattern, result)):  # stop generating once the concatenated LLM output matches the 
            # RETRIEVE("some query") format
            query = m.group('query')
            logger.info(f'The LLM generated a valid query: {repr(result)} -> {repr(query)}')
            return query

    logger.info(f'The LLM did not generate a valid query: {repr(result)}')
    # the last user input is used as a query
    logger.info(f'Using the user input as query: {repr(user_msg)}')
    return user_msg


def prompt_with_retrieval(
    model: TextGenerator,
    input: str,  # full content of the playground's text window (conversation with "User:" and "Assistant:" prompts)
    config: dict,  # decoding config
    retriever_config: Optional[dict] = None,
) -> tuple[str, Optional[str]]:
    """
    Decides whether to retrieve, retrieves documents and append them to the input then returns the new input.
    Also returns text that should be shown to the user (e.g., the titles of the retrieved documents).
    """
    # If the prompt ends with a retriever query, use it for retrieval instead of the user's message or generating one
    retriever_query = None
    retriever_query_index = input.rfind('\nRetriever query:')
    if retriever_query_index >= input.rfind('\nUser:'):
        retriever_query = input[retriever_query_index:].removeprefix('\nRetriever query:').strip()

    # Remove retriever queries from the prompt as they might confuse the model
    input = regex.sub(r'\nRetriever query:.*?(\n|$)', r'\1', input, flags=regex.DOTALL)

    split_input = regex.split('\nUser:', input)
    user_msg = split_input[-1].strip() if len(split_input) > 1 else None
    history = input.removesuffix(user_msg)  # remove the user message from the prompt as it can be put somewhere else
    # by the prompt template (e.g., after or before the retrieved documents or both). Note that `history` should end 
    # with "User:"
    history = history.lstrip('\n')

    if not retriever_query:
        # Decide whether to retrieve and use the model to generate a query for the retriever:
        retriever_query = generate_query(model, history, user_msg, config, retriever_config)
    if not retriever_query:  # skip retrieval
        return input, None

    logger.info(f'Retrieving from {retriever_config}')
    docs = retrieve(retriever_query, retriever_config)
    retriever_output_template = retriever_config.get('output_template') or ''
    if '{user_msg}' not in retriever_output_template:  # add the {user_msg} placeholder if missing
        retriever_output_template = '{user_msg}' + '\n' + retriever_output_template
    
    # For example:
    """
    How old is he?
    You can use the following documents to guide your answer:
     
    {docs}

    Given the above documents, please answer the following query as faithfully as possible: How old is he?
    """

    if '{docs}' not in retriever_output_template:  # add the {docs} placeholder if missing (will be replaced by 
        # the content of retrieved documents)
        retriever_output_template += '\n' + '{docs}'
    
    if docs:
        max_length = retriever_config['max_doc_len']  # total max length
        min_length = retriever_config['min_doc_len']  # min length of each document
        topk = retriever_config['topk']
        length = 0
        docs_truncated = []
        for doc in docs:
            text = doc['formatted']
            # tokenize the documents to measure their length and skip those that are too short or too long
            tokenized_text = model.task.preprocessor.tokenize(text)
            doc_len = len(tokenized_text)
            remaining_len = max_length - length
            if remaining_len < min_length:
                break
            if doc_len < min_length:
                continue
            if doc_len > remaining_len:  # skip this document and try to find one that fits, but we could truncate
                # it instead like this:
                # tokenized_text = tokenized_text[:remaining_len]
                # logger.info(f'Last document was truncated to: {len(tokenized_text)}/{doc_len}')
                # doc['formatted'] = model.task.preprocessor.detokenize(tokenized_text)
                continue
            length += len(tokenized_text)
            docs_truncated.append(doc)
            doc['formatted'] = f'# Document {len(docs_truncated)}\n' + doc['formatted']  # these extra tokens are not 
            # counted in `doc_len`, so `max_length` should leave some margin w.r.t. the model's max context length
            if length >= max_length:
                break
        docs = docs_truncated[:topk]
        if docs:
            docs_str = '\n\n'.join(doc['formatted'] for doc in docs)
            new_input = history + retriever_output_template.format(user_msg=user_msg, docs=docs_str)
            # For example:
            """
            User: Who is the prime minister of France?
            Assistant: Gabriel Attal
            User: How old is he?
            You can use the following documents to guide your answer:
            
            # Document 1
            ## Title: Gabriel Attal
            Attal was born on 16 March 1989 in Clamart, Île-de-France. [...]

            Given the above documents, please answer the following query as faithfully as possible: How old is he?
            """
            return new_input, {
                'retriever_query': retriever_query,
                'retrieved_docs': docs,
            }
        else:
            logger.info('Retrieved documents are too long')
    else:
        logger.info('No document found')

    return input, None  # no retrieval, `input` is not modified


def chat(model: TextGenerator, input: str, config: dict, retriever_config: Optional[dict] = None):
    """
    Like `TextGenerator.stream()` but with some dialogue-specific features: allows retrieval-augmented generation
    using the last user's message as query; and automatically adds the "User:" and "Assistant:" prompts.
    """
    default_retriever_config = get_retriever_config(model.model_info)
    retriever_config = retriever_config or {}
    retriever_config = {**default_retriever_config, **retriever_config}
    strip_whitespace = False
    input = '\n' + input  # for "\nUser:" to always match
    if input.rfind('\nUser:') >= input.rfind('\nAssistant:'):  # if the last message in the conversation is a user
        # message, then generate the assistant's answer (possibly with retrieval). If not, just continue the 
        # previous assistant message.
        input, output = prompt_with_retrieval(model, input, config, retriever_config)  # optionally use retrieval to 
        # modify the LLM's prompt (with retrieved documents)
        if output:
            yield output
        yield {'detok': '\nAssistant: '}
        strip_whitespace = True
        input = input + '\nAssistant:'
    input = input.lstrip('\n')

    for out in model.stream(input, **config):
        if strip_whitespace:  # some models start their assistant answers by a whitespace prefix, this is redundant
            # with the whitespace we add after the "Assistant:" prompt.
            out['detok'] = out['detok'].lstrip()
            strip_whitespace = (not out['detok'])  # continue stripping whitespaces as long as we don't generate
            # something else
        yield out
    yield {'detok': '\nUser: '}


def round_all(object, ndigits=5):
    return utils.apply(
        functools.partial(round, ndigits=ndigits),
        object,
        Number,
    )


@app.route('/<name>/stream', methods=['GET'])
def stream(name):
    """
    Generate outputs word by word (not compatible with beam search and with batching)
    """
    config = get_data()
    config['beam_size'] = 1
    logger.info('input: %s', config)
    model = get_model(name)
    input = process_input(config.pop('input', None))  # 'input' should be removed from config, because it is passed
    # as a positional argument to `TextGenerator.stream()`
    if not input:
        input = ['']
    if len(input) > 1:
        abort(400, 'batching is not compatible with streaming, use /decode')
    # normalize the prompt to remove unnecessary whitespaces
    input = input[0]
    logger.warning(input.encode())
    chat_mode = model.model_info['task'] == 'dialogue'
    retriever_config = config.pop('retriever_config', None)

    output = []
    with utils.suppress():
        if chat_mode:  # if the model is a dialogue model. This is adds missing prompts ("Assistant:" prompt 
            # if the last message is a user message, and "User:" prompt at the end of the assistant's answer) and 
            # supports retrieval-augmented generation.
            output = chat(model, input, config, retriever_config=retriever_config)
        else:
            output = model.stream(input, **config)
    output = itertools.chain(output, [{}])
    
    output = (f'data: {json.dumps(round_all(out))}\n\n' for out in output)
    return Response(output, mimetype='text/event-stream')



@app.route('/<name>/decode', methods=['GET', 'POST'])
def decode(name):
    """
    Run batched decoding: stop generating once EOS or max length is reached and return the full output.
    Contrary to `stream()`, this is compatible with batches and with beam search.
    """
    config = get_data()
    
    logger.info('input: %s', config)
    model = get_model(name)

    input = process_input(config.pop('input', None))
    start = time.time()
    output = model.decode(*input, **config, return_scores=True)
    elapsed = time.time() - start

    # Only get the top-1 hypotheses
    output = [nbest[0] for nbest in output]
    # Convert a list of dicts into a dict of list
    # While the list format is convenient, we need to send other things that are not lists (e.g., elapsed time)
    if output:
        output = {
            k: [hyp[k] for hyp in output]
            for k in output[0]
        }
    output = utils.array_to_list(output)
    output.update({
        'elapsed': elapsed,
        'model': name,
    })
    output['detok'] = [
        line.strip().removesuffix('\\n').replace('\\n', '\n')
        for line in output['detok']
    ]
    output['input'] = input  # should this be normalized?

    output = round_all(output)  # to avoid overly long numbers in the json output
    logger.info('output: %s', output)
    return jsonify(output)



def main():
    global retriever_url
    args = parser.parse_args()
    retriever_url = args.retriever_url
    
    mp.set_start_method("spawn")

    for model_path in args.models:
        model_path = model_path.rstrip('/')
        model_path, *device = model_path.rsplit(':', maxsplit=1)
        opts = {}
        opts['devices'] = parse_devices(device[0] if device else None)
        if args.dtype:
            opts['dtype'] = args.dtype
        model = TextGenerator.build(model_path, **opts)
        model_name = model.model_info['name']
        models[model_name] = model
    serve(app, host="0.0.0.0", port=args.port)

if __name__ == '__main__':
    main()
