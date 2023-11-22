#!/usr/bin/env bash
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

# Download English-French OpenSubtitles, Europarl, TED2020 and News-commentary, which are all document-level 
# ordered corpora

DATA_DIR=data/Doc-level

mkdir -p ${DATA_DIR}/raw

pushd ${DATA_DIR}/raw

wget https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-fr.txt.zip -O OpenSubtitles.en-fr.txt.zip
unzip OpenSubtitles.en-fr.txt.zip
wget https://opus.nlpl.eu/download.php?f=TED2020/v1/moses/en-fr.txt.zip -O TED2020.en-fr.txt.zip
unzip TED2020.en-fr.txt.zip 

wget https://data.statmt.org/news-commentary/v16/training/news-commentary-v16.en-fr.tsv.gz
gunzip news-commentary-v16.en-fr.tsv.gz
cut -f1 news-commentary-v16.en-fr.tsv > news-commentary.en-fr.en
cut -f2 news-commentary-v16.en-fr.tsv > news-commentary.en-fr.fr

wget https://www.statmt.org/europarl/v10/training/europarl-v10.fr-en.tsv.gz
gunzip europarl-v10.fr-en.tsv.gz
cut -f1 europarl-v10.fr-en.tsv > europarl.en-fr.fr
cut -f2 europarl-v10.fr-en.tsv > europarl.en-fr.en

popd

# filter the corpora to remove any line pair in the wrong language, normalize the whitespaces and remove empty lines
for corpus in news-commentary europarl TED2020 OpenSubtitles; do
    scripts/filter-corpus.py ${DATA_DIR}/raw/${corpus}.en-fr.{en,fr} \
    --actions clean langid -o ${DATA_DIR}/${corpus}.en-fr.{en,fr} -v
done

cp examples/ParaCrawl/fr-en/{dict.txt,bpecodes} ${DATA_DIR}

sacrebleu --download wmt13 -l en-fr
examples/Doc-level-MT/xml2doc.py < ~/.sacrebleu/wmt13/raw/test/newstest2013-src.en.sgm --seg-tag --skip-xml | examples/Doc-level-MT/doc2sent.py --context 2 > ${DATA_DIR}/newstest2013.en-fr.en
examples/Doc-level-MT/xml2doc.py < ~/.sacrebleu/wmt13/raw/test/newstest2013-src.fr.sgm --seg-tag --skip-xml | examples/Doc-level-MT/doc2sent.py --context 2 > ${DATA_DIR}/newstest2013.en-fr.fr
sacrebleu --download wmt14 -l en-fr
examples/Doc-level-MT/xml2doc.py < ~/.sacrebleu/wmt14/raw/test/newstest2014-fren-ref.en.sgm --seg-tag --skip-xml | examples/Doc-level-MT/doc2sent.py --context 2 > ${DATA_DIR}/newstest2014.en-fr.en
examples/Doc-level-MT/xml2doc.py < ~/.sacrebleu/wmt14/raw/test/newstest2014-fren-src.fr.sgm --seg-tag --skip-xml | examples/Doc-level-MT/doc2sent.py --context 2 > ${DATA_DIR}/newstest2014.en-fr.fr
sacrebleu --download wmt15 -l en-fr
examples/Doc-level-MT/xml2doc.py < ~/.sacrebleu/wmt15/raw/test/newsdiscusstest2015-enfr-src.en.sgm --seg-tag --skip-xml | examples/Doc-level-MT/doc2sent.py --context 2 > ${DATA_DIR}/newsdiscusstest2015.en-fr.en
examples/Doc-level-MT/xml2doc.py < ~/.sacrebleu/wmt15/raw/test/newsdiscusstest2015-enfr-ref.fr.sgm --seg-tag --skip-xml | examples/Doc-level-MT/doc2sent.py --context 2 > ${DATA_DIR}/newsdiscusstest2015.en-fr.fr

examples/download-flores.sh
cp data/FLORES/FLORES-valid.{en,fr} ${DATA_DIR}
