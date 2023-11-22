#!/usr/bin/env bash
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

set -e

DATA_DIR=data/TED

mkdir -p ${DATA_DIR}/archives

pushd ${DATA_DIR}

wget -qnc http://phontron.com/data/ted_talks.tar.gz -P archives
tar xzf archives/ted_talks.tar.gz

wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/deescape-special-chars.perl

function clean {
    perl deescape-special-chars.perl | sed 's/.*NULL.*//' | python -c "import sys; sys.stdout.writelines(' '.join(line.split()) + '\n' for line in sys.stdin)"
}

langs=`head -n1 all_talks_train.tsv`

tail -n+2 all_talks_train.tsv | shuf > all_talks_train.shuf.tsv
mv all_talks_train.shuf.tsv all_talks_train.tsv

i=1
for lang in ${langs}; do
    cat all_talks_train.tsv | cut -f${i} | clean > train.${lang}
    tail -n+2 all_talks_dev.tsv | cut -f${i} | clean > valid.${lang}
    tail -n+2 all_talks_test.tsv | cut -f${i} | clean > test.${lang}
    i=$((i + 1))
done

rm {train,valid,test}.talk_name
rm {train,valid,test}.calv
rm all_talks_{train,dev,test}.tsv

for corpus in train valid test; do
    mv ${corpus}.{pt-br,pt_br}
    mv ${corpus}.{fr-ca,fr_ca}
    mv ${corpus}.{zh-cn,zh_cn}
    mv ${corpus}.{zh-tw,zh_tw}
done

langs=$( echo $langs | sed 's/-/_/g' | sed 's/calv\|talk_name//g' )

for corpus in valid test; do
    for lang in ${langs}; do
        if [ ${lang} != en ]; then
            paste ${corpus}.${lang} ${corpus}.en | grep -Pv "^\t|\t$" | cut -f1 > ${corpus}.${lang}-en.${lang}
            paste ${corpus}.${lang} ${corpus}.en | grep -Pv "^\t|\t$" | cut -f2 > ${corpus}.${lang}-en.en
        fi
    done
done

popd

# copy existing tokenizers (de-en and top20)
cp -r examples/TED/de-en ${DATA_DIR}
ln -rs ${DATA_DIR}/{de-en,en-de}
mkdir -p ${DATA_DIR}/top20
cp examples/TED-top20/{dict.txt,bpecodes} ${DATA_DIR}/top20
