#!/usr/bin/bash
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

set -e

MODEL=$1

DATA_DIR=data/iwslt2023

mkdir -p ${DATA_DIR}

# Download speech feature models
pushd ${DATA_DIR}
if [ "${MODEL}" = w2v2nima ]; then
    if [ ! -d "${MODEL}" ]; then
        git lfs install
        git clone https://huggingface.co/LIA-AvignonUniversity/IWSLT2022-Niger-Mali ${MODEL}
    fi
    LAYER=8
elif [ "${MODEL}" = xlsr128 ]; then
    if [ ! -d "${MODEL}" ]; then
        git lfs install
        git clone https://huggingface.co/facebook/wav2vec2-xls-r-300m ${MODEL}
    fi
    LAYER=18
else
    echo "Usage: $0 xlsr128|w2v2nima" >&2
    exit 1
fi
popd

MODEL_DIR=${DATA_DIR}/${MODEL}
FEAT_DIR=${MODEL_DIR}-${LAYER}

# Preprocess Tamasheq data
mkdir -p ${FEAT_DIR}/tamasheq
for split in train valid test; do
    npy_output=${FEAT_DIR}/tamasheq/${split}.npy.taq
    txt_output=${FEAT_DIR}/tamasheq/${split}.fr

    if [ ! -s ${npy_output} ]; then
        echo "### Preprocessing Tamasheq ${split}"
        cut -f1,3,4 -d' ' ${DATA_DIR}/IWSLT2022_Tamasheq_data/taq_fra_clean/${split}/txt/segments | \
        examples/IWSLT2023/extract-features.py ${MODEL_DIR} --layer-id ${LAYER} \
        --audio-dirs ${DATA_DIR}/IWSLT2022_Tamasheq_data/taq_fra_clean/${split} \
        --txt-file ${DATA_DIR}/IWSLT2022_Tamasheq_data/taq_fra_clean/${split}/txt/${split}.fra \
        -o ${npy_output} > ${txt_output}
    fi
done

# Preprocess Quechua data
mkdir -p ${FEAT_DIR}/quechua
for split in train valid; do
    npy_output=${FEAT_DIR}/quechua/${split}.npy.que
    txt_output=${FEAT_DIR}/quechua/${split}.es

    if [ ! -s ${npy_output} ]; then
        echo "### Preprocessing Quechua ${split}"
        cat ${DATA_DIR}/IWSLT2023_Quechua_data/que_spa_clean/${split}/txt/segments | \
        python3 -c "import sys; lines = [line.split() for line in sys.stdin]; sys.stdout.writelines(f'{name}\t{start}\t{end}\n' for name, *_, start, end in lines)" | \
        examples/IWSLT2023/extract-features.py ${MODEL_DIR} --layer-id ${LAYER} \
        --audio-dirs ${DATA_DIR}/IWSLT2023_Quechua_data/que_spa_clean/${split} \
        --txt-file ${DATA_DIR}/IWSLT2023_Quechua_data/que_spa_clean/${split}/txt/${split}.spa \
        -o ${npy_output} > ${txt_output}
    fi
done

# Preprocess mTEDx data
for pair in es-en es-es es-fr es-it es-pt fr-en fr-es fr-fr fr-pt it-en it-es it-it pt-en pt-es pt-pt; do
    src=$( echo $pair | cut -d'-' -f1 )
    tgt=$( echo $pair | cut -d'-' -f2 )
    
    mkdir -p ${FEAT_DIR}/mtedx/${pair}
    for split in train valid test iwslt2021; do
        npy_output=${FEAT_DIR}/mtedx/${pair}/${split}.npy.${src}
        txt_output=${FEAT_DIR}/mtedx/${pair}/${split}.${tgt}

        if [[ ! -s ${npy_output} ]]; then
            echo "### Preprocessing mTEDx ${pair}/${split}"
            cut -f2,3,4 -d' ' ${DATA_DIR}/mtedx/${pair}/data/${split}/txt/segments | \
            examples/IWSLT2023/extract-features.py ${MODEL_DIR} --layer-id ${LAYER} \
            --audio-dirs ${DATA_DIR}/mtedx/${pair}/data/${split}/wav --file-extension flac \
            --txt-file ${DATA_DIR}/mtedx/${pair}/data/${split}/txt/${split}.${tgt} \
            -o ${npy_output} > ${txt_output}
        fi
    done
done

# Preprocess TED-LIUM data
mkdir -p ${FEAT_DIR}/ted-lium
for split in train dev test; do
    if [ ${split} = dev ]; then
        npy_output=${FEAT_DIR}/ted-lium/valid.npy.en
        txt_output=${FEAT_DIR}/ted-lium/valid.en
    else
        npy_output=${FEAT_DIR}/ted-lium/${split}.npy.en
        txt_output=${FEAT_DIR}/ted-lium/${split}.en
    fi

    if [[ ! -s ${npy_output} ]]; then
        echo "### Preprocessing TED-LIUM ${split}"

        cat ${DATA_DIR}/TEDLIUM_release2/${split}/stm/*.stm | grep -v ignore_time_segment_in_scoring \
        > ${DATA_DIR}/TEDLIUM_release2/${split}.txt
        
        cat ${DATA_DIR}/TEDLIUM_release2/${split}/stm/*.stm | grep -v ignore_time_segment_in_scoring | cut -d' ' -f1,4,5 \
        > ${DATA_DIR}/TEDLIUM_release2/${split}.segments
        
        examples/IWSLT2023/extract-features.py ${MODEL_DIR} --layer-id ${LAYER} \
        -i ${DATA_DIR}/TEDLIUM_release2/${split}.segments \
        --audio-dirs ${DATA_DIR}/TEDLIUM_release2/${split}/sph --file-extension wav \
        --txt-file ${DATA_DIR}/TEDLIUM_release2/${split}.txt \
        -o ${npy_output} > ${txt_output}
    fi
done