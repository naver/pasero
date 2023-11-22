# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import sys
import os
import regex
import traceback
import sacrebleu
import logging
import collections
import multiprocessing
import functools
import itertools
import random
from typing import Iterable, Iterator, Sequence, Optional

METRICS = ['chrf', 'bleu', 'langid', 'len_ratio', 'chrf++', 'spbleu', 'wer']
BLEU_TOKENIZERS = sacrebleu.metrics.METRICS['BLEU'].TOKENIZERS


def lower_is_better(metric: str) -> bool:
    return 'loss' in metric or 'ppl' in metric


def langid_py(line: str) -> str:
    import langid
    return langid.classify(line)[0]


def langid_accuracy(hyps: list[str], refs: list[str], model: str = 'lid218') -> float:
    if model == 'lid218':
        import fasttext
        path = download('https://tinyurl.com/nllblid218e', 'lid218e.bin')
        model = fasttext.load_model(path)
        langs = sum(model.predict(list(hyps) + list(refs))[0], [])
        langs = [lang.removeprefix('__label__') for lang in langs]
    elif model =='langid_py':
        p = multiprocessing.Pool()
        langs = p.map(langid_py, list(hyps) + list(refs))
    else:
        raise NotImplementedError
    hyp_langs = langs[:len(hyps)]
    ref_langs = langs[len(hyps):]
    # guess the expected language by looking for the most frequent language in refs
    lang, total = collections.Counter(ref_langs).most_common(1)[0]
    # only count pairs where both the hypothesis and reference are in the expected language
    correct = sum(ref_lang == hyp_lang == lang for hyp_lang, ref_lang in zip(hyp_langs, ref_langs))
    return correct / total if total > 0 else 0.0


def clean_spaces(line: str) -> str:
    return ' '.join(line.split())


class LineReader:
    def __init__(self, *paths: str, start: Optional[int] = None, stop: Optional[int] = None):
        self.paths = paths
        self.start = start
        self.stop = stop
        self.total = 0
    
    def __iter__(self) -> Iterator[tuple[int, tuple[str, ...]]]:
        files = [open(path) for path in self.paths]
        for tuple in itertools.islice(enumerate(zip(*files)), self.start, self.stop):
            self.total += 1
            yield tuple


class ParallelCleaner:
    def __init__(self, line_tuples_with_index: Iterable[tuple[int, tuple[str, ...]]]) -> None:
        self.line_tuples_with_index = line_tuples_with_index
        self.total = 0
        self.correct = 0

    @classmethod
    def read_files(cls, *paths, **kwargs) -> Iterator[tuple[int, tuple[str, ...]]]:
        return cls(LineReader(*paths), **kwargs)

    @classmethod
    def filter(cls, line_tuples: Iterable[tuple[str, ...]], **kwargs) -> Iterator[tuple[str, ...]]:
        for _, line_tuple in cls(enumerate(line_tuples), **kwargs):
            yield line_tuple

    def __iter__(self) -> Iterator[tuple[int, tuple[str, ...]]]:
        raise NotImplementedError

    @property
    def ratio(self):
        return self.correct / max(self.total, 1)


class FilterByLang(ParallelCleaner):
    def __init__(
        self,
        line_tuples_with_index: Iterable[tuple[int, tuple[str, ...]]],
        langs: Sequence[str],
        chunksize: int = 10**3,
    ):
        super().__init__(line_tuples_with_index)
        self.langs = langs
        path = download('https://tinyurl.com/nllblid218e', 'lid218e.bin')
        import fasttext
        self.model = fasttext.load_model(path)
        self.chunksize = chunksize

    def __iter__(self) -> Iterator[tuple[int, tuple[str, ...]]]:
        it = iter(self.line_tuples_with_index)
        while True:
            chunk = list(itertools.islice(it, self.chunksize))
            if not chunk:
                break
            n = len(chunk[0])
            assert n == len(self.langs)
            flattened = []
            indices = []
            for i, line_tuple in chunk:
                flattened += [clean_spaces(line) for line in line_tuple]
                indices.append(i)
            pred = sum(self.model.predict(flattened)[0], [])
            pred = [lang.removeprefix('__label__') for lang in pred]
            chunk = [flattened[i*n:(i + 1)*n] for i in range(len(flattened)//n)]
            pred = [pred[i*n:(i + 1)*n] for i in range(len(pred)//n)]
            for i, lines, pred_ in zip(indices, chunk, pred):
                self.total += 1
                if all(lines) and list(pred_) == list(self.langs):
                    self.correct += 1
                    lines = tuple(lines)
                    yield i, tuple(lines)


class Dedup(ParallelCleaner):
    def __init__(
        self,
        line_tuples_with_index: Iterable[tuple[int, tuple[str, ...]]],
        lc: bool = False,
        monolingual: bool = False,
        no_punct: bool = False,
    ):
        super().__init__(line_tuples_with_index)
        self.lc = lc
        self.monolingual = monolingual
        self.no_punct = no_punct

    def __iter__(self) -> Iterator[tuple[int, tuple[str, ...]]]:
        seen = set()
        for index, lines in self.line_tuples_with_index:
            self.total += 1
            lines = [clean_spaces(line) for line in lines]
            if not all(lines):
                continue

            normalized = []
            for line in lines:
                if self.lc:
                    line = line.lower()
                if self.no_punct:
                    line = clean_spaces(regex.sub('\W', ' ', line))
                normalized.append(line)

            keys = normalized if self.monolingual else [tuple(normalized)]
            if any(key in seen for key in keys):
                continue
            seen.update(keys)
            self.correct += 1
            yield index, lines


class FilterByLen(ParallelCleaner):
    def __init__(
        self,
        line_tuples_with_index: Iterable[tuple[int, tuple[str, ...]]],
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
        max_ratio: Optional[float] = None,
        char_level: bool = False,
        byte_level: bool = False,
        length_correction: bool = False,
        langs: Optional[Sequence[str]] = None,
    ):
        if length_correction:
            assert char_level and langs
        super().__init__(line_tuples_with_index)
        self.min_len, self.max_len = min_len, max_len
        self.max_ratio = max_ratio
        self.char_level = char_level
        self.byte_level = byte_level
        self.length_correction = length_correction
        self.langs = langs

    # language-specific length correction factors described in the NLLB-200 paper (= chars(en)/chars(lang))
    len_ratios = {'ace_Arab': 1.18, 'ace_Latn': 0.93, 'acm_Arab': 1.16, 'acq_Arab': 1.15, 'aeb_Arab': 1.17, 'afr_Latn': 0.94, 'ajp_Arab': 1.21, 'aka_Latn': 1.0, 'amh_Ethi': 1.47, 'apc_Arab': 1.21, 'arb_Arab': 1.13, 'ars_Arab': 1.13, 'ary_Arab': 1.16, 'arz_Arab': 1.16, 'asm_Beng': 1.03, 'ast_Latn': 0.97, 'awa_Deva': 1.02, 'ayr_Latn': 0.95, 'azb_Arab': 1.13, 'azj_Latn': 0.92, 'bak_Cyrl': 0.98, 'bam_Latn': 1.04, 'ban_Latn': 0.89, 'bel_Cyrl': 0.87, 'bem_Latn': 0.81, 'ben_Beng': 1.02, 'bho_Deva': 1.03, 'bjn_Arab': 1.08, 'bjn_Latn': 0.95, 'bod_Tibt': 0.89, 'bos_Latn': 0.99, 'bug_Latn': 0.93, 'bul_Cyrl': 0.96, 'cat_Latn': 0.91, 'ceb_Latn': 0.83, 'ces_Latn': 1.03, 'cjk_Latn': 0.92, 'ckb_Arab': 1.03, 'crh_Latn': 0.97, 'cym_Latn': 0.94, 'dan_Latn': 0.97, 'deu_Latn': 0.85, 'dik_Latn': 1.16, 'dyu_Latn': 0.99, 'dzo_Tibt': 0.79, 'ell_Grek': 0.83, 'eng_Latn': 1.0, 'epo_Latn': 1.0, 'est_Latn': 1.02, 'eus_Latn': 0.94, 'ewe_Latn': 1.02, 'fao_Latn': 0.98, 'pes_Arab': 1.05, 'fij_Latn': 0.86, 'fin_Latn': 0.93, 'fon_Latn': 0.98, 'fra_Latn': 0.84, 'fur_Latn': 0.91, 'fuv_Latn': 1.06, 'gla_Latn': 0.81, 'gle_Latn': 0.86, 'glg_Latn': 0.9, 'grn_Latn': 0.99, 'guj_Gujr': 1.04, 'hat_Latn': 1.08, 'hau_Latn': 0.93, 'heb_Hebr': 1.28, 'hin_Deva': 1.0, 'hne_Deva': 1.03, 'hrv_Latn': 1.01, 'hun_Latn': 0.95, 'hye_Armn': 0.9, 'ibo_Latn': 0.99, 'ilo_Latn': 0.83, 'ind_Latn': 0.92, 'isl_Latn': 1.01, 'ita_Latn': 0.85, 'jav_Latn': 0.96, 'jpn_Jpan': 2.29, 'kab_Latn': 1.01, 'kac_Latn': 0.79, 'kam_Latn': 1.02, 'kan_Knda': 0.95, 'kas_Arab': 1.04, 'kas_Deva': 1.04, 'kat_Geor': 0.91, 'knc_Arab': 1.13, 'knc_Latn': 0.95, 'kaz_Cyrl': 0.97, 'kbp_Latn': 0.91, 'kea_Latn': 1.01, 'khm_Khmr': 0.83, 'kik_Latn': 0.85, 'kin_Latn': 0.89, 'kir_Cyrl': 0.98, 'kmb_Latn': 0.9, 'kon_Latn': 0.88, 'kor_Hang': 1.99, 'kmr_Latn': 1.0, 'lao_Laoo': 1.0, 'lvs_Latn': 0.98, 'lij_Latn': 0.91, 'lim_Latn': 0.97, 'lin_Latn': 0.92, 'lit_Latn': 0.99, 'lmo_Latn': 0.93, 'ltg_Latn': 1.01, 'ltz_Latn': 0.89, 'lua_Latn': 0.93, 'lug_Latn': 0.98, 'luo_Latn': 0.95, 'lus_Latn': 0.92, 'mag_Deva': 1.04, 'mai_Deva': 1.02, 'mal_Mlym': 0.88, 'mar_Deva': 1.0, 'min_Latn': 0.94, 'mkd_Cyrl': 0.96, 'plt_Latn': 0.81, 'mlt_Latn': 0.91, 'mni_Beng': 0.97, 'khk_Cyrl': 0.95, 'mos_Latn': 1.04, 'mri_Latn': 0.9, 'zsm_Latn': 0.89, 'mya_Mymr': 0.8, 'nld_Latn': 0.9, 'nno_Latn': 0.98, 'nob_Latn': 0.99, 'npi_Deva': 1.04, 'nso_Latn': 0.87, 'nus_Latn': 0.93, 'nya_Latn': 0.89, 'oci_Latn': 0.88, 'gaz_Latn': 0.84, 'ory_Orya': 0.97, 'pag_Latn': 0.99, 'pan_Guru': 0.99, 'pap_Latn': 0.95, 'pol_Latn': 0.94, 'por_Latn': 0.92, 'prs_Arab': 1.09, 'pbt_Arab': 1.02, 'quy_Latn': 0.94, 'ron_Latn': 0.89, 'run_Latn': 0.9, 'rus_Cyrl': 0.91, 'sag_Latn': 0.92, 'san_Deva': 1.01, 'sat_Beng': 0.94, 'scn_Latn': 0.94, 'shn_Mymr': 0.7, 'sin_Sinh': 0.98, 'slk_Latn': 0.99, 'slv_Latn': 1.0, 'smo_Latn': 0.85, 'sna_Latn': 0.89, 'snd_Arab': 1.1, 'som_Latn': 0.88, 'sot_Latn': 0.83, 'spa_Latn': 0.84, 'als_Latn': 0.89, 'srd_Latn': 0.86, 'srp_Cyrl': 1.01, 'ssw_Latn': 0.89, 'sun_Latn': 0.95, 'swe_Latn': 0.99, 'swh_Latn': 0.95, 'szl_Latn': 0.96, 'tam_Taml': 0.86, 'tat_Cyrl': 0.98, 'tel_Telu': 0.98, 'tgk_Cyrl': 0.9, 'tgl_Latn': 0.79, 'tha_Thai': 1.04, 'tir_Ethi': 1.45, 'taq_Latn': 1.05, 'taq_Tfng': 1.06, 'tpi_Latn': 0.78, 'tsn_Latn': 0.79, 'tso_Latn': 0.83, 'tuk_Latn': 0.94, 'tum_Latn': 0.77, 'tur_Latn': 0.97, 'twi_Latn': 1.02, 'tzm_Tfng': 1.12, 'uig_Arab': 0.94, 'ukr_Cyrl': 0.97, 'umb_Latn': 0.99, 'urd_Arab': 1.0, 'uzn_Latn': 0.88, 'vec_Latn': 0.99, 'vie_Latn': 0.95, 'war_Latn': 0.8, 'wol_Latn': 1.04, 'xho_Latn': 0.94, 'ydd_Hebr': 0.93, 'yor_Latn': 1.02, 'yue_Hant': 3.29, 'zho_Hans': 2.97, 'zho_Hant': 3.2, 'zul_Latn': 0.89}

    def __iter__(self) -> Iterator[tuple[int, tuple[str, ...]]]:
        for index, lines in self.line_tuples_with_index:
            self.total += 1
            lines = [clean_spaces(line) for line in lines]
            if not all(lines):
                continue
            if self.char_level:
                lengths = [len(line) for line in lines]
                if self.length_correction:
                    assert len(lengths) == len(self.langs)
                    lengths = [length * FilterByLen.len_ratios[lang] for length, lang in zip(lengths, self.langs)]
            elif self.byte_level:
                lengths = [len(line.encode()) for line in lines]
            else:
                lengths = [len(line.split()) for line in lines]
            if self.min_len and min(lengths) < self.min_len:
                continue
            if self.max_len and max(lengths) > self.max_len:
                continue
            if self.max_ratio and max(lengths) / min(lengths) > self.max_ratio:
                continue
            self.correct += 1
            yield index, lines



class Shuffle(ParallelCleaner):
    def __init__(
        self,
        line_tuples_with_index: Iterable[tuple[int, tuple[str, ...]]],
        seed: int = 1234,
    ):
        super().__init__(line_tuples_with_index)
        self.seed = seed
    
    def __iter__(self) -> Iterator[tuple[int, tuple[str, ...]]]:
        lines = list(self.line_tuples_with_index)
        self.correct = self.total = len(lines)
        random.seed(self.seed)
        random.shuffle(lines)
        yield from lines


filter_by_lang = FilterByLang.filter
dedup = Dedup.filter
filter_by_len = FilterByLen.filter
shuffle = Shuffle.filter


def download(url: str, filename: str) -> str:
    tmp_dir = os.environ.get('PASERO_TMP') or 'tmp'
    path = os.path.join(tmp_dir, filename)
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(tmp_dir, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def score(
    metric: str,
    hyps: list[str],
    refs: Optional[list[str]],
    bleu_tok: Optional[str] = None,
    eval_lc: bool = False,
    silent: bool = True,
) -> Optional[float]:
    assert refs is None or len(refs) == len(hyps), "different number of references and hypotheses"

    try:
        if silent:
            # disable logging for, as some metrics libraries can be very verbose
            devnull = open(os.devnull, 'w')
            stderr = sys.stderr
            stdout = sys.stdout
            sys.stderr = devnull
            sys.stdout = devnull
            logging.disable(logging.CRITICAL)

        if eval_lc:
            hyps = [hyp.lower() for hyp in hyps]
            refs = [ref.lower() for ref in refs] if refs else refs
        chrf_word_order = 2 if metric == 'chrf++' else 0
        bleu_tok = 'flores200' if metric == 'spbleu' else bleu_tok

        if metric == 'wer':
            from jiwer import wer
            hyps, refs = zip(*[(hyp, ref) for hyp, ref in zip(hyps, refs) if hyp and ref])
            score = 100 * wer(list(refs), list(hyps))
        elif metric in ('bleu', 'spbleu'):
            metric_ = sacrebleu.metrics.BLEU(tokenize=bleu_tok, force=True)
            out = metric_.corpus_score(hyps, [refs])
            score = out.score
        elif metric in ('chrf', 'chrf++'):
            metric_ = sacrebleu.metrics.CHRF(word_order=chrf_word_order)
            out = metric_.corpus_score(hyps, [refs])
            score = out.score
        elif metric == 'len_ratio':
            score = sacrebleu.corpus_bleu(hyps, [refs], tokenize=bleu_tok, force=True).ratio
        elif metric == 'langid':
            score = langid_accuracy(hyps, refs)
        else:
            raise NotImplementedError(f"unsupported metric '{metric}'")
    finally:
        if silent:
            sys.stderr = stderr
            sys.stdout = stdout
            logging.disable(logging.NOTSET)

    return score


def safe_score(*args, **kwargs) -> Optional[float]:
    # Version of score() that won't raise exceptions (just print them)
    try:
        return score(*args, **kwargs)
    except NotImplementedError:
        pass
    except:
        traceback.print_exc()
    return 0


def score_file(
    metric: str,
    hyp_path: Optional[str],
    ref_path: Optional[str],
    bleu_tok: Optional[str] = None,
    eval_lc: bool = False,
    strict: bool = True,
):
    hyp_file = sys.stdin if not hyp_path or hyp_path == '-' else open(hyp_path)
    hyps = [line.strip() for line in hyp_file]
    refs = None if ref_path is None else [line.strip() for line in open(ref_path)]
    if not strict and hyps and refs:
        hyps = hyps[:len(refs)]
        refs = refs[:len(hyps)]
    return score(metric, hyps, refs, bleu_tok=bleu_tok, eval_lc=eval_lc)


def score_files(
    metric: str,
    hyp_paths: list[str],
    ref_paths: list[str],
    bleu_tok: Optional[str] = None,
    eval_lc: bool = False,
    strict: bool = True,
):
    """ Score multiple files in parallel """
    from multiprocessing import Pool
    assert len(hyp_paths) == len(ref_paths) and len(hyp_paths) >= 1
    p = Pool()
    score_ = functools.partial(score_file, bleu_tok=bleu_tok, eval_lc=eval_lc, strict=strict)
    metric = [metric] * len(hyp_paths)
    args = list(zip(metric, hyp_paths, ref_paths))
    return p.starmap(score_, args)
