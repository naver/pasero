#!/usr/bin/env bash
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

# Download ParaCrawl in 25 languages paired with English and aligns these bilingual corpora (by pivoting through English)
# to create a multi-parallel corpus.
# Note that this download and pre-processing is very long and memory-intensive.
# If you are only interested in a single bilingual ParaCrawl corpus (e.g., French-English), prefer using
# `examples/ParaCrawl/download.sh LANG`

DATA_DIR=data/ParaCrawl-Euro
mkdir -p ${DATA_DIR}/archives
ALL_LANGS=( fr de es it pt nl nb cs pl sv da el fi hr hu bg ro sk lt lv sl et ga is mt )

# Download & clean the data

for lang in ${ALL_LANGS[@]}; do
    URL=https://s3.amazonaws.com/web-language-models/paracrawl/release9/en-${lang}/en-${lang}.txt.gz
    F=${DATA_DIR}/archives/ParaCrawl.en-${lang}.txt.gz
    echo "# Downloading ${lang} corpus"
    wget -qnc ${URL} -O ${F}
done


for lang in ${ALL_LANGS[@]}; do
    PAIR=${lang}-en
    F=${DATA_DIR}/archives/ParaCrawl.en-${lang}.txt.gz
    src=${DATA_DIR}/archives/ParaCrawl.${PAIR}.${lang}
    tgt=${DATA_DIR}/archives/ParaCrawl.${PAIR}.en

    if [ -f ${src} ] && [ -f ${tgt} ]; then
        continue
    fi

    {
        echo "# Unzipping ${lang} corpus"
        gunzip < ${F} > ${DATA_DIR}/archives/ParaCrawl.${PAIR}
        # Normalize whitespaces
        cut -f1 ${DATA_DIR}/archives/ParaCrawl.${PAIR} | python3 -c "import sys; sys.stdout.writelines(' '.join(line.split()) + '\n' for line in sys.stdin)" > ${tgt}
        cut -f2 ${DATA_DIR}/archives/ParaCrawl.${PAIR} | python3 -c "import sys; sys.stdout.writelines(' '.join(line.split()) + '\n' for line in sys.stdin)" > ${src}
        rm ${DATA_DIR}/archives/ParaCrawl.${PAIR}
    } &
done
wait

mkdir -p ${DATA_DIR}/multiparallel

echo "# Building multi-parallel corpus"
if [ ! -f ${DATA_DIR}/multiparallel/ParaCrawl.en ]; then
    # create a single file containing all the unique English lines
    for lang in ${ALL_LANGS[@]}; do
        cat ${DATA_DIR}/archives/ParaCrawl.${lang}-en.en
    done | python3 -c "import sys; sys.stdout.writelines(dict.fromkeys(sys.stdin))" > ${DATA_DIR}/multiparallel/ParaCrawl.en
fi

for lang in ${ALL_LANGS[@]}; do
    PAIR=${lang}-en
    echo ${PAIR}
    src=${DATA_DIR}/archives/ParaCrawl.${PAIR}.${lang}
    tgt=${DATA_DIR}/archives/ParaCrawl.${PAIR}.en

    if [ -f ${DATA_DIR}/multiparallel/ParaCrawl.${lang} ]; then
        continue
    fi

    # create a multi-parallel corpus where each language is aligned with the English file: lines that do not have a translation are left empty
    cat ${DATA_DIR}/multiparallel/ParaCrawl.en | \
    python3 -c "import sys; d = {}; [d.__setitem__(tgt, src) for src, tgt in zip(open('${src}'), open('${tgt}')) if tgt not in d]; sys.stdout.writelines(d.get(line, '\n') for line in sys.stdin)" > ${DATA_DIR}/multiparallel/ParaCrawl.${lang}
done

mkdir -p ${DATA_DIR}/bilingual

for src in ${ALL_LANGS[@]}; do
    for tgt in ${ALL_LANGS[@]}; do
        prefix=${DATA_DIR}/bilingual/ParaCrawl
        corpus=${prefix}.${src}-${tgt}

        if [ -s ${corpus}.${src} ] && [ -s ${corpus}.${tgt} ]; then
            continue
        fi

        if [ ${src} = ${tgt} ]; then
            continue
        elif [[ "${src}" < "${tgt}" ]]; then
            {
                echo "# Building ${src}-${tgt} bilingual corpus"
                # Using the multiparallel corpus (with N aligned files), build N*(N-1) bilingual corpora (by removing empty lines & deduplicating) and shuffle them
                paste ${DATA_DIR}/multiparallel/ParaCrawl.${src} ${DATA_DIR}/multiparallel/ParaCrawl.${tgt} | grep -Pv "^\t|\t$" | \
                python3 -c "import sys, random; seen_src = set(); seen_tgt = set(); lines = [seen_src.add(src) or seen_tgt.add(tgt) or f'{src}\t{tgt}\n' for src, tgt in (line.rstrip('\n').split('\t') for line in sys.stdin) if src not in seen_src and tgt not in seen_tgt]; random.seed(42); random.shuffle(lines); sys.stdout.writelines(lines)" > ${corpus}
                cut -f1 ${corpus} > ${corpus}.${src}
                cut -f2 ${corpus} > ${corpus}.${tgt}
                rm ${corpus}
            } &
        else
            ln -frs ${prefix}.${tgt}-${src}.${src} ${corpus}.${src}
            ln -frs ${prefix}.${tgt}-${src}.${tgt} ${corpus}.${tgt}
        fi
    done
    wait
done

for src in ${ALL_LANGS[@]}; do
    tgt=en
    prefix=${DATA_DIR}/bilingual/ParaCrawl
    corpus=${prefix}.${src}-${tgt}

    if [ -f ${corpus}.${src} ] && [ -f ${corpus}.${tgt} ]; then
        continue
    fi

    echo "# Building ${src}-${tgt} bilingual corpus"
    paste ${DATA_DIR}/multiparallel/ParaCrawl.${src} ${DATA_DIR}/multiparallel/ParaCrawl.${tgt} | grep -Pv "^\t|\t$" | \
    python3 -c "import sys, random; seen_src = set(); seen_tgt = set(); lines = [seen_src.add(src) or seen_tgt.add(tgt) or f'{src}\t{tgt}\n' for src, tgt in (line.rstrip('\n').split('\t') for line in sys.stdin) if src not in seen_src and tgt not in seen_tgt]; random.seed(42); random.shuffle(lines); sys.stdout.writelines(lines)" > ${corpus}
    cut -f1 ${corpus} > ${corpus}.${src}
    cut -f2 ${corpus} > ${corpus}.${tgt}
    rm ${corpus}
    ln -frs ${corpus}.${src} ${prefix}.${tgt}-${src}.${src}
    ln -frs ${corpus}.${tgt} ${prefix}.${tgt}-${src}.${tgt}
done

# Download the FLORES-200 valid and test sets
examples/download-flores.sh

mkdir -p data/FLORES/euro
for corpus in FLORES-valid FLORES-test; do
    rm -f data/FLORES/euro/${corpus}.*
    for src in en ${ALL_LANGS[@]}; do
        head -n 100 data/FLORES/${corpus}.${src} >> data/FLORES/euro/${corpus}.src
        for tgt in en ${ALL_LANGS[@]}; do
            head -n 100 data/FLORES/${corpus}.${tgt} >> data/FLORES/euro/${corpus}.${tgt}
        done
    done
done

# copy existing tokenizers
cp examples/ParaCrawl-Euro/{dict.txt,bpecodes} ${DATA_DIR}
