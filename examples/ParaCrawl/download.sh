#!/usr/bin/env bash
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

# Download a ParaCrawl corpus for this language paired with English. For instance:
# `examples/ParaCrawl/download.sh fr` will download a French-English parallel corpus as
# 'data/ParaCrawl/ParaCrawl.en-fr.en' and 'data/ParaCrawl/ParaCrawl.en-fr.fr'

if [ $# -lt 1 ]; then
    echo "Usage: $0 LANG [MAX_LINES]" >&2
    exit 1
fi

LANG=$1
PAIR=en-$LANG
MAX_LINES=$2

# This isn't the same as "examples/ParaCrawl-Euro/download.sh" which creates a multi-parallel corpus in 26 languages
# from English-centric ParaCrawl. It also creates a bilingual ParaCrawl French-English corpus, but smaller because of 
# more aggressive deduplication.
DATA_DIR=data/ParaCrawl

mkdir -p ${DATA_DIR}

pushd ${DATA_DIR}

URL=https://web-language-models.s3.us-east-1.amazonaws.com/paracrawl/release9/$PAIR/$PAIR.txt.gz

if [ -z "${MAX_LINES}" ]; then
    wget ${URL}  # can take a few hours
    gunzip $PAIR.txt.gz
else
    wget ${URL} -O - | gunzip | head -n ${MAX_LINES} > ${PAIR}.txt
fi

cut -f1 $PAIR.txt > ParaCrawl.$PAIR.en
cut -f2 $PAIR.txt > ParaCrawl.$PAIR.$LANG
rm $PAIR.txt
popd

# Download the FLORES-200 valid and test sets
examples/download-flores.sh

# copy existing tokenizers
if [ ${LANG} = "fr" ] && [ ! -d "${DATA_DIR}/fr-en" ]; then
    cp -r examples/ParaCrawl/fr-en ${DATA_DIR}
    ln -rs ${DATA_DIR}/{fr-en,en-fr}
fi