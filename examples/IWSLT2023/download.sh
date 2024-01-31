#!/usr/bin/env bash
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

if ! command -v sox &> /dev/null
then
    echo "sox is not installed (required to convert audio files)"
    exit 1
fi

mkdir -p data/iwslt2023

pushd data/iwslt2023

wget https://www.openslr.org/resources/100/mtedx_iwslt2021.tgz
tar xzf mtedx_iwslt2021.tgz
mv mtedx_iwslt2021 mtedx

for lang in es-en es-fr es-it es-pt fr-en fr-es fr-pt pt-en pt-es it-en it-es fr pt es it; do
    wget https://www.openslr.org/resources/100/mtedx_${lang}.tgz
    tar xzf mtedx_${lang}.tgz -C mtedx
done

wget --no-check-certificate https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release2.tar.gz
tar xzf TEDLIUM_release2.tar.gz 
for f in TEDLIUM_release2/*/sph/*.sph; do sox -t sph "$f" -r 16000 -c 1 -b 16 -e signed-integer "${f%.*}.wav"; done

git clone https://github.com/mzboito/IWSLT2022_Tamasheq_data.git
git clone https://github.com/Llamacha/IWSLT2023_Quechua_data.git

# download the NLLB model and tokenizer files
wget --trust-server-names https://tinyurl.com/flores200sacrebleuspm -O spm.model
wget --trust-server-names https://tinyurl.com/nllb200dictionary -O dict.txt
wget --trust-server-names https://tinyurl.com/nllb200densedst1bcheckpoint -O nllb_1.3B_distilled.bin

# modify NLLB's dictionaries to add the language codes, but using 2-letter language codes when known (following the same format as mTEDx)
for lang in ace_Arab ace_Latn acm_Arab acq_Arab aeb_Arab af ajp_Arab aka_Latn am apc_Arab ar ars_Arab ary_Arab arz_Arab asm_Beng ast awa_Deva ayr_Latn azb_Arab az ba bam_Latn ban_Latn be bem_Latn bn bho_Deva bjn_Arab bjn_Latn bod_Tibt bs bug_Latn bg ca ceb cs cjk_Latn ckb_Arab crh_Latn cy da de dik_Latn dyu_Latn dzo_Tibt el en epo_Latn et eus_Latn ewe_Latn fao_Latn fa fij_Latn fi fon_Latn fr fur_Latn ff gd ga gl grn_Latn gu ht ha he hi hne_Deva hr hu hy ig ilo id is it jv ja kab_Latn kac_Latn kam_Latn kn kas_Arab kas_Deva ka knc_Arab knc_Latn kk kbp_Latn kea_Latn km kik_Latn kin_Latn kir_Cyrl kmb_Latn kon_Latn ko kmr_Latn lo lv lij_Latn lim_Latn ln lt lmo_Latn ltg_Latn lb lua_Latn lg luo_Latn lus_Latn mag_Deva mai_Deva ml mr min_Latn mk mg mlt_Latn mni_Beng mn mos_Latn mri_Latn ms my nl nno_Latn no ne ns nus_Latn nya_Latn oc gaz_Latn or pag_Latn pa pap_Latn pl pt prs_Arab ps quy_Latn ro run_Latn ru sag_Latn san_Deva sat_Beng scn_Latn shn_Mymr si sk sl smo_Latn sna_Latn sd so sot_Latn es sq srd_Latn sr ss su sv sw szl_Latn ta tat_Cyrl tel_Telu tgk_Cyrl tl th tir_Ethi taq_Latn taq_Tfng tpi_Latn tn tso_Latn tuk_Latn tum_Latn tr twi_Latn tzm_Tfng uig_Arab uk umb_Latn ur uz vec_Latn vi war_Latn wo xh yi yo yue_Hant zh zho_Hant zu; do
    echo "<lang:${lang}> 0" >> dict.txt
done
echo "<mask>" >> dict.txt  # use dummy token position for mask token, in case we want to do denoising
for token in madeupword0001 madeupword0002; do
    echo "${token} 0" >> dict.txt
done

popd
