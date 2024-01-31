# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import sys
import regex
import json
import itertools
import unicodedata
import functools
import logging
import time
import copy
import io
import multiprocessing
import numpy as np
from collections import Counter, defaultdict
from typing import Iterable, Iterator, Optional, Union
from .noise import mask


logger = logging.getLogger('tokenizer')


def inline_case_to_cased(tokens: list[str]) -> list[str]:
    tokens = list(tokens)
    for i, w in enumerate(tokens):
        if w == '<T>':
            tokens[i - 1] = tokens[i - 1].title()
        elif w == '<U>':
            tokens[i - 1] = tokens[i - 1].upper()
    return [w for w in tokens if w not in ('<T>', '<U>')]


def detokenize(tokens: list[str], strip: bool = True) -> str:
    tokens = inline_case_to_cased(tokens)
    x = ' '.join(w for w in tokens if w != '</s>')
    x = x.replace(' ', '').replace('▁', ' ')
    return x.strip() if strip else x


scripts = {
    'Arabic': [(1547, 1547), (1549, 1557), (1566, 1566), (1569, 1594), (1601, 1610), (1622, 1630), (1642, 1647), (1649, 1756), (1758, 1791), (1872, 1901), (64336, 64433), (64467, 64829), (64848, 64911), (64914, 64967), (65008, 65020), (65136, 65140), (65142, 65276)],
    'Armenian': [(1329, 1366), (1369, 1375), (1377, 1415), (1418, 1418), (64275, 64279)],
    'Bengali': [(2433, 2435), (2437, 2444), (2447, 2448), (2451, 2472), (2474, 2480), (2482, 2482), (2486, 2489), (2492, 2500), (2503, 2504), (2507, 2510), (2519, 2519), (2524, 2525), (2527, 2531), (2534, 2554)],
    'Bopomofo': [(12549, 12588), (12704, 12727)],
    'Braille': [(10240, 10495)],
    'Buginese': [(6656, 6683), (6686, 6687)],
    'Buhid': [(5952, 5971)],
    'Canadian_Aboriginal': [(5121, 5750)],
    'Cherokee': [(5024, 5108)],
    'Common': [(0, 64), (91, 96), (123, 169), (171, 185), (187, 191), (215, 215), (247, 247), (697, 735), (741, 767), (894, 894), (903, 903), (1417, 1417), (1536, 1539), (1548, 1548), (1563, 1563), (1567, 1567), (1600, 1600), (1632, 1641), (1757, 1757), (2404, 2405), (2416, 2416), (3647, 3647), (4347, 4347), (5867, 5869), (5941, 5942), (8192, 8203), (8206, 8291), (8298, 8304), (8308, 8318), (8320, 8334), (8352, 8373), (8448, 8485), (8487, 8489), (8492, 8524), (8531, 8579), (8592, 9179), (9216, 9254), (9280, 9290), (9312, 9884), (9888, 9905), (9985, 9988), (9990, 9993), (9996, 10023), (10025, 10059), (10061, 10061), (10063, 10066), (10070, 10070), (10072, 10078), (10081, 10132), (10136, 10159), (10161, 10174), (10176, 10182), (10192, 10219), (10224, 10239), (10496, 11027), (11776, 11799), (11804, 11805), (12272, 12283), (12288, 12292), (12294, 12294), (12296, 12320), (12336, 12343), (12348, 12351), (12443, 12444), (12448, 12448), (12539, 12539), (12688, 12703), (12736, 12751), (12832, 12867), (12880, 12895), (12926, 13054), (13056, 13311), (19904, 19967), (42752, 42774), (57344, 63743), (64830, 64831), (65021, 65021), (65040, 65049), (65072, 65106), (65108, 65126), (65128, 65131), (65279, 65279), (65281, 65312), (65339, 65344), (65371, 65381), (65392, 65392), (65438, 65439), (65504, 65510), (65512, 65518), (65529, 65533), (65792, 65794), (65799, 65843), (65847, 65855), (118784, 119029), (119040, 119078), (119082, 119142), (119146, 119162), (119171, 119172), (119180, 119209), (119214, 119261), (119552, 119638), (119808, 119892), (119894, 119964), (119966, 119967), (119970, 119970), (119973, 119974), (119977, 119980), (119982, 119993), (119995, 119995), (119997, 120003), (120005, 120069), (120071, 120074), (120077, 120084), (120086, 120092), (120094, 120121), (120123, 120126), (120128, 120132), (120134, 120134), (120138, 120144), (120146, 120485), (120488, 120777), (120782, 120831), (917505, 917505), (917536, 917631), (983040, 1048573)],
    'Coptic': [(994, 1007), (11392, 11498), (11513, 11519)],
    'Cypriot': [(67584, 67589), (67592, 67592), (67594, 67637), (67639, 67640), (67644, 67644), (67647, 67647)],
    'Cyrillic': [(1024, 1158), (1160, 1230), (1232, 1273), (1280, 1295), (7467, 7467), (7544, 7544)],
    'Deseret': [(66560, 66639)],
    'Devanagari': [(2305, 2361), (2364, 2381), (2384, 2388), (2392, 2403), (2406, 2415), (2429, 2429)],
    'Ethiopic': [(4608, 4680), (4682, 4685), (4688, 4694), (4696, 4696), (4698, 4701), (4704, 4744), (4746, 4749), (4752, 4784), (4786, 4789), (4792, 4798), (4800, 4800), (4802, 4805), (4808, 4822), (4824, 4880), (4882, 4885), (4888, 4954), (4959, 4988), (4992, 5017), (11648, 11670), (11680, 11686), (11688, 11694), (11696, 11702), (11704, 11710), (11712, 11718), (11720, 11726), (11728, 11734), (11736, 11742)],
    'Georgian': [(4256, 4293), (4304, 4346), (4348, 4348), (11520, 11557)],
    'Glagolitic': [(11264, 11310), (11312, 11358)],
    'Gothic': [(66352, 66378)],
    'Greek': [(884, 885), (890, 890), (900, 902), (904, 906), (908, 908), (910, 929), (931, 974), (976, 993), (1008, 1023), (7462, 7466), (7517, 7521), (7526, 7530), (7936, 7957), (7960, 7965), (7968, 8005), (8008, 8013), (8016, 8023), (8025, 8025), (8027, 8027), (8029, 8029), (8031, 8061), (8064, 8116), (8118, 8132), (8134, 8147), (8150, 8155), (8157, 8175), (8178, 8180), (8182, 8190), (8486, 8486), (65856, 65930), (119296, 119365)],
    'Gujarati': [(2689, 2691), (2693, 2701), (2703, 2705), (2707, 2728), (2730, 2736), (2738, 2739), (2741, 2745), (2748, 2757), (2759, 2761), (2763, 2765), (2768, 2768), (2784, 2787), (2790, 2799), (2801, 2801)],
    'Gurmukhi': [(2561, 2563), (2565, 2570), (2575, 2576), (2579, 2600), (2602, 2608), (2610, 2611), (2613, 2614), (2616, 2617), (2620, 2620), (2622, 2626), (2631, 2632), (2635, 2637), (2649, 2652), (2654, 2654), (2662, 2676)],
    'Han': [(11904, 11929), (11931, 12019), (12032, 12245), (12293, 12293), (12295, 12295), (12321, 12329), (12344, 12347), (12353, 12438), (12445, 12447), (12449, 12538), (12540, 12543), (12784, 12799), (13312, 19893), (19968, 40891), (63744, 64045), (64048, 64106), (64112, 64217), (65382, 65391), (65393, 65437), (131072, 173782), (194560, 195101)],
    'Hangul': [(4352, 4441), (4447, 4514), (4520, 4601), (12593, 12686), (12800, 12830), (12896, 12925), (44032, 55203), (65440, 65470), (65474, 65479), (65482, 65487), (65490, 65495), (65498, 65500)],
    'Hanunoo': [(5920, 5940)],
    'Hebrew': [(1425, 1465), (1467, 1479), (1488, 1514), (1520, 1524), (64285, 64310), (64312, 64316), (64318, 64318), (64320, 64321), (64323, 64324), (64326, 64335)],
    'Inherited': [(768, 879), (1611, 1621), (1648, 1648), (7616, 7619), (8204, 8205), (8400, 8427), (12330, 12335), (12441, 12442), (65024, 65039), (65056, 65059), (119143, 119145), (119163, 119170), (119173, 119179), (119210, 119213), (917760, 917999)],
    'Kannada': [(3202, 3203), (3205, 3212), (3214, 3216), (3218, 3240), (3242, 3251), (3253, 3257), (3260, 3268), (3270, 3272), (3274, 3277), (3285, 3286), (3294, 3294), (3296, 3297), (3302, 3311)],
    'Kharoshthi': [(68096, 68099), (68101, 68102), (68108, 68115), (68117, 68119), (68121, 68147), (68152, 68154), (68159, 68167), (68176, 68184)],
    'Khmer': [(6016, 6109), (6112, 6121), (6128, 6137), (6624, 6655)],
    'Lao': [(3713, 3714), (3716, 3716), (3719, 3720), (3722, 3722), (3725, 3725), (3732, 3735), (3737, 3743), (3745, 3747), (3749, 3749), (3751, 3751), (3754, 3755), (3757, 3769), (3771, 3773), (3776, 3780), (3782, 3782), (3784, 3789), (3792, 3801), (3804, 3805)],
    'Latin': [(65, 90), (97, 122), (170, 170), (186, 186), (192, 214), (216, 246), (248, 577), (592, 696), (736, 740), (7424, 7461), (7468, 7516), (7522, 7525), (7531, 7543), (7545, 7615), (7680, 7835), (7840, 7929), (8305, 8305), (8319, 8319), (8336, 8340), (8490, 8491), (64256, 64262), (65313, 65338), (65345, 65370)],
    'Limbu': [(6400, 6428), (6432, 6443), (6448, 6459), (6464, 6464), (6468, 6479)],
    'Linear_B': [(65536, 65547), (65549, 65574), (65576, 65594), (65596, 65597), (65599, 65613), (65616, 65629), (65664, 65786)],
    'Malayalam': [(3330, 3331), (3333, 3340), (3342, 3344), (3346, 3368), (3370, 3385), (3390, 3395), (3398, 3400), (3402, 3405), (3415, 3415), (3424, 3425), (3430, 3439)],
    'Mongolian': [(6144, 6158), (6160, 6169), (6176, 6263), (6272, 6313)],
    'Myanmar': [(4096, 4129), (4131, 4135), (4137, 4138), (4140, 4146), (4150, 4153), (4160, 4185)],
    'New_Tai_Lue': [(6528, 6569), (6576, 6601), (6608, 6617), (6622, 6623)],
    'Ogham': [(5760, 5788)],
    'Old_Italic': [(66304, 66334), (66336, 66339)],
    'Old_Persian': [(66464, 66499), (66504, 66517)],
    'Oriya': [(2817, 2819), (2821, 2828), (2831, 2832), (2835, 2856), (2858, 2864), (2866, 2867), (2869, 2873), (2876, 2883), (2887, 2888), (2891, 2893), (2902, 2903), (2908, 2909), (2911, 2913), (2918, 2929)],
    'Osmanya': [(66688, 66717), (66720, 66729)],
    'Runic': [(5792, 5866), (5870, 5872)],
    'Shavian': [(66640, 66687)],
    'Sinhala': [(3458, 3459), (3461, 3478), (3482, 3505), (3507, 3515), (3517, 3517), (3520, 3526), (3530, 3530), (3535, 3540), (3542, 3542), (3544, 3551), (3570, 3572)],
    'Syloti_Nagri': [(43008, 43051)],
    'Syriac': [(1792, 1805), (1807, 1866), (1869, 1871)],
    'Tagalog': [(5888, 5900), (5902, 5908)],
    'Tagbanwa': [(5984, 5996), (5998, 6000), (6002, 6003)],
    'Tai_Le': [(6480, 6509), (6512, 6516)],
    'Tamil': [(2946, 2947), (2949, 2954), (2958, 2960), (2962, 2965), (2969, 2970), (2972, 2972), (2974, 2975), (2979, 2980), (2984, 2986), (2990, 3001), (3006, 3010), (3014, 3016), (3018, 3021), (3031, 3031), (3046, 3066)],
    'Telugu': [(3073, 3075), (3077, 3084), (3086, 3088), (3090, 3112), (3114, 3123), (3125, 3129), (3134, 3140), (3142, 3144), (3146, 3149), (3157, 3158), (3168, 3169), (3174, 3183)],
    'Thaana': [(1920, 1969)],
    'Thai': [(3585, 3642), (3648, 3675)],
    'Tibetan': [(3840, 3911), (3913, 3946), (3953, 3979), (3984, 3991), (3993, 4028), (4030, 4044), (4047, 4049)],
    'Tifinagh': [(11568, 11621), (11631, 11631)],
    'Ugaritic': [(66432, 66461), (66463, 66463)],
    'Yi': [(40960, 42124), (42128, 42182)],
}
scripts = {
    i: script.lower() for script, positions in scripts.items()
    for start, end in positions
    for i in range(start, end + 1)
}
scripts = [scripts.get(i) for i in range(max(scripts) + 1)]     # maps unicode code point to script name

def get_script(s):
    i = ord(s[0])
    return scripts[i] if i < len(scripts) else None

script_ids = {script: i for i, script in enumerate(list(set(scripts)), 1)}
script_ids = [script_ids.get(script, 0) for script in scripts]  # maps unicode code point to script id
def get_script_id(s):
    i = ord(s[0])
    return script_ids[i] if i < len(script_ids) else None

def split_by_script_(tokens):
    new_tokens = []
    for token in tokens:
        cur_script = None
        cur_token = ''
        for x in token:
            script = get_script_id(x)
            if cur_script is not None and x != ' ' and x != '▁' and script != cur_script:
                new_tokens.append(cur_token)
                cur_token = ''
            cur_token += x
            if x != ' ' and x != '▁':
                cur_script = script
        if cur_token:
            new_tokens.append(cur_token)
    return new_tokens


_PROTECT_SYMBOL = '╳'
_MASK_SYMBOL = '⧈'
_PHL_SYMBOL = '⧇'
_WHITESPACE_REGEX = regex.compile(r'\s+')
_NO_MIXED_CASE_REGEX = regex.compile('(▁?[[:upper:]]?[^[:upper:]\s▁{0}]+|▁?[[:upper:]]+|▁|{0})'.format(
    regex.escape(_PROTECT_SYMBOL)))
_SENTENCEPIECE_REGEX = regex.compile('(▁?[^\s▁{0}]+|▁|{0})'.format(regex.escape(_PROTECT_SYMBOL)))

_TOKENIZATION_REGEXES = [
    None,                                                              # 0
    regex.compile(r'(▁?[[:alnum:]]+|[^[:alnum:]]+)'),                  # 1
    regex.compile(r'(▁?[[:alpha:]]+|▁?[[:digit:]]+|[^[:alnum:]]+)'),   # 2
    regex.compile(r'(▁?[[:alpha:]]+|▁?[[:digit:]]+|[^[:alnum:]])'),    # 3
    regex.compile(r'(▁?[[:alpha:]]+|▁?[[:digit:]]|[^[:alnum:]])'),     # 4
]
_UPPER_CODE, _TITLE_CODE, _LOWER_CODE = range(3)
_CASE_SYMBOLS = ['<U>', '<T>', None]


class PaseroTokenizer:
    """
    This is a modified version of subword-nmt (https://github.com/rsennrich/subword-nmt)
    """
    def __init__(
        self,
        path_or_merges: Union[str, list[tuple[str, str]]],
        vocab: Optional[list[str]] = None,
        inline_case: bool = True,
        nfkc: bool = False,
        protect_regex: Optional[str] = None,
        **kwargs,
    ):
        config = dict(kwargs)

        if isinstance(path_or_merges, str):
            with open(path_or_merges) as bpe_file:
                first_line = next(bpe_file)
                if first_line.startswith('#'):
                    try:
                        config = json.loads(first_line.strip('# \n\r'))
                    except:
                        pass
                else:
                    bpe_file = itertools.chain([first_line], bpe_file)
            
                merges = [tuple(line.rstrip('\r\n').rsplit(' ', maxsplit=1)) for line in bpe_file]
        else:
            merges = list(path_or_merges)
        
        self.inline_case = inline_case
        self.nfkc = nfkc
        self.protect_regex = protect_regex

        for key in 'inline_case', 'protect_regex', 'nfkc':
            if key in config:
                setattr(self, key, config[key])

        self.merges = {code: i for i, code in reversed(list(enumerate(merges)))}
        self.merges_reverse = {a + b: (a, b) for a, b in self.merges}
        self.vocab = set(vocab) if vocab else None
        if self.protect_regex is not None:
            self.protect_regex = regex.compile(self.protect_regex)
        self.cache = {}
        self.chars = None

    @classmethod
    def train(cls, inputs, output=None, num_symbols=8000, verbose=False, threads=None, existing_bpe_path=False,
              **kwargs):
        """
        Learn num_symbols BPE operations from vocabulary and write to output
        """
        start = time.time()
        vocabs, line_counts = cls._get_vocabularies(inputs, threads=threads, **kwargs)
        vocab = cls._merge_vocabularies(vocabs, line_counts, **kwargs)

        chars = {}
        for w, c in vocab.items():
            for x in w:
                chars[x] = chars.get(x, 0) + c
        chars = dict(sorted(chars.items(), key=lambda item: item[1], reverse=True))

        vocab_time = round(time.time() - start, 1)
        if verbose:
            logger.info(f'finished reading vocabulary in {vocab_time}s: {len(vocab)} unique tokens')

        start = time.time()
        config = {
            key: kwargs[key] for key in ('tokenization', 'inline_case', 'protect_regex', 'nfkc')
            if key in kwargs
        }
        
        if existing_bpe_path:
            bpe_model = cls(existing_bpe_path)
            merges = list(reversed(bpe_model.merges))
        else:
            merges = []

        if output is None:
            outfile = sys.stdout
        else:
            dirname = os.path.dirname(output)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            outfile = open(output, 'w')
        
        print('#', json.dumps(config, ensure_ascii=False), file=outfile)

        vocab = dict([(tuple(x), y) for x, y in vocab.items()])

        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

        stats, indices = cls._get_pair_statistics(sorted_vocab)
        big_stats = copy.deepcopy(stats)
        # threshold is inspired by Zipfian assumption, but should only affect speed
        threshold = max(stats.values()) / 10

        def key(stats, pair):
            return stats[pair], pair

        while not existing_bpe_path:
            i = len(merges)
            if i >= num_symbols:
                break

            if stats:
                most_frequent = max(stats, key=lambda x: key(stats, x))

            # we probably missed the best pair because of pruning; go back to full statistics
            if not stats or i and stats[most_frequent] < threshold:
                cls._prune_stats(stats, big_stats, threshold)
                stats = copy.deepcopy(big_stats)
                most_frequent = max(stats, key=lambda x: key(stats, x))
                # threshold is inspired by Zipfian assumption, but should only affect speed
                threshold = stats[most_frequent] * i / (i + 10000)
                cls._prune_stats(stats, big_stats, threshold)

            if stats[most_frequent] < 2:
                break

            if verbose:
                logger.info('pair {0}: {1} {2} -> {1}{2} (frequency {3})'.format(i + 1, most_frequent[0],
                    most_frequent[1], stats[most_frequent]))

            print(*most_frequent, file=outfile)
            merges.append(most_frequent)

            cls._update_pair_statistics(most_frequent, sorted_vocab, stats, indices)
            stats[most_frequent] = 0
            if not i % 100:
                cls._prune_stats(stats, big_stats, threshold)

        bpe_time = round(time.time() - start, 1)
        if verbose:
            logger.info(f'finished building the BPE model in {bpe_time}s')
        start = time.time()

        bpe_model = cls(merges)
        # encode the vocabularies with the BPE model. The result can be used to generate a dictionary that maps
        # existing tokens to their frequency
        pool = multiprocessing.Pool(processes=threads) if threads is None or threads > 1 else None
        vocabs = {k: v for k, v in zip(
            vocabs.keys(),
            pool.map(bpe_model._encode_vocab, vocabs.values())
        )}

        encode_time = round(time.time() - start, 1)
        if verbose:
            logger.info(f'finished encoding the vocabularies in {encode_time}s '
                        f'(total {round(bpe_time + vocab_time + encode_time, 1)}s)')

        return bpe_model, vocabs

    def get_vocab(self, inputs, max_lines=10**7):
        vocab = Counter()
        for filename in inputs:
            with open(filename) as infile:
                lines = self._readlines(infile, max_lines)
                vocab_ = Counter(token for line in lines for token in self._tokenize(line).split())
                for symbol in _CASE_SYMBOLS:
                    if symbol in vocab_:
                        vocab_.pop(symbol)
                read_bytes = infile.tell()
                infile.seek(0, io.SEEK_END)
                total_bytes = infile.tell()
                r = total_bytes / read_bytes
                for k in vocab_:
                    vocab_[k] = int(vocab_[k] * r)
                vocab += vocab_
        return vocab

    def __len__(self):
        return len(self.merges)

    def tokenize(
        self,
        sentence: str,
        unk: Optional[str] = None,
        dropout: float = 0.0,
        spell_out: float = 0.0,
    ) -> list[str]:
        tokens = self._tokenize(sentence, unk=unk, dropout=dropout, spell_out=spell_out)
        if tokens and tokens[0] == '▁':
            # a lone meta symbol at the beginning of a sentence serves no purpose
            tokens.pop(0)
        return tokens

    def _tokenize(
        self,
        sentence: str,
        unk: Optional[str] = None,
        dropout: float = 0.0,
        spell_out: float = 0.0,
    ) -> list[str]:
        sentence = sentence.strip()

        if not sentence:
            return []

        if self.nfkc:
            sentence = unicodedata.normalize('NFKC', sentence)

        if self.protect_regex is not None:
            sentence = sentence.replace(_PROTECT_SYMBOL, ' ')
            protected_tokens = [
                m.group(0) for m in self.protect_regex.finditer(sentence)
            ]
            sentence = self.protect_regex.sub(_PROTECT_SYMBOL, sentence)

        # protect mask & placeholder tokens against BPE tokenization by replacing them with this weird character
        sentence = sentence.replace(_MASK_SYMBOL, '').replace(mask, _MASK_SYMBOL)
        sentence = sentence.replace(_PHL_SYMBOL, '').replace('<PHL>', _PHL_SYMBOL)

        if self.inline_case:
            for symbol in _CASE_SYMBOLS:
                if symbol is not None:
                    sentence = sentence.replace(symbol, ' ')

        sentence = sentence.replace('▁', ' ')
        sentence = '▁' + _WHITESPACE_REGEX.sub('▁', sentence)
        if self.inline_case:
            tokens = _NO_MIXED_CASE_REGEX.findall(sentence)
        else:
            tokens = _SENTENCEPIECE_REGEX.findall(sentence)

        if self.inline_case:
            cased_tokens = tokens
            tokens = [token.lower() for token in tokens]

        wordpieces = [
            [] if not word else self._encode_word_cached(word, dropout=dropout, spell_out=spell_out)
            for word in tokens
        ]

        if self.inline_case:
            wordpieces_ = []
            for cased_token, wordpiece in zip(cased_tokens, wordpieces):
                i = 0
                wordpiece_ = []
                for out in wordpiece:
                    x = cased_token[i:i + len(out)]
                    i += len(out)
                    if x.isupper():
                        wordpiece_.append((out, _UPPER_CODE))
                    elif x.istitle():
                        wordpiece_.append((out, _TITLE_CODE))
                    else:
                        wordpiece_.append((out, _LOWER_CODE))
                wordpieces_.append(wordpiece_)

            wordpieces = [
                ' '.join(self._add_factor(token, case) for (token, case) in wordpiece)
                for wordpiece in wordpieces_
            ]
        else:
            wordpieces = [' '.join(wordpiece) for wordpiece in wordpieces]

        sentence = ' '.join(wordpieces)

        if self.protect_regex is not None:
            sentence = sentence.replace(_PROTECT_SYMBOL + ' ▁ ', _PROTECT_SYMBOL + ' ')
            for token in protected_tokens:
                sentence = sentence.replace(_PROTECT_SYMBOL, token, 1)
            sentence = _WHITESPACE_REGEX.sub(' ', sentence)
        
        sentence = sentence.replace(_MASK_SYMBOL, mask)
        sentence = sentence.replace(_PHL_SYMBOL, '<PHL>')

        tokens = sentence.split()
        if unk is not None and self.vocab:
            tokens = [w if w in self.vocab else unk.replace('{token}', w) for w in tokens]
        return tokens

    def _add_factor(self, token, case):
        if self.inline_case:
            case_symbol = _CASE_SYMBOLS[case]
            if case_symbol is not None:
                token += ' ' + case_symbol
        return token

    def _encode_word(self, word, dropout=0.0):
        word = list(word)

        while len(word) > 1:
            pairs = list(dict.fromkeys(pair for pair in zip(word, word[1:]) if pair in self.merges))
            # using dict instead of set, because set has a non-deterministic order

            if dropout:
                pairs = [pair for pair in pairs if np.random.random() > dropout]

            if not pairs:
                break

            bigram = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
            if bigram not in self.merges:
                break
            left, right = bigram

            new_word = []
            skip = False
            for x, y in zip(word, word[1:]):
                if skip:
                    skip = False
                    continue

                if x == left and y == right:
                    new_word.append(x + y)
                    skip = True
                else:
                    new_word.append(x)
            if not skip:
                new_word.append(y)

            word = new_word

        return [x for item in word for x in self._recursive_split(item)]

    def _recursive_split(self, segment):
        if self.vocab is None or segment in self.vocab or segment not in self.merges_reverse:
            yield segment
        else:
            for item in self.merges_reverse[segment]:
                yield from self._recursive_split(item)

    def _encode_word_cached(self, word, dropout=0.0, spell_out=0.0):
        # simple LRU cache implementation (functools.lru_cache is not pickable)
        if spell_out and np.random.random() < spell_out:
            return list(word)
        elif dropout:
            return self._encode_word(word, dropout=dropout)
        elif word in self.cache:
            new_word = self.cache.pop(word)
            self.cache[word] = new_word   # put this entry last in the cache
            return new_word
        else:
            new_word = self._encode_word(word)
            self.cache[word] = new_word
            if len(self.cache) > 2**20:
                word = next(iter(self.cache.keys()))   # delete first (oldest) entry
                self.cache.pop(word)
            return new_word

    def _encode_vocab(self, vocab):
        new_vocab = defaultdict(int)
        for word, count in vocab.items():
            for token in self._encode_word(word):
                new_vocab[token] += count
        return Counter(new_vocab)

    @staticmethod
    def _get_vocabulary(buffer, tokenization=2, inline_case=True, split_by_script=True, nfkc=False, delimiter=None,
                        protect_regex=None, **_):
        vocab = {}
        line_count = 0
        for line in buffer:
            line = line.strip()

            if not line:
                continue

            # not counting empty lines, important for temperature-based oversampling over
            # multi-aligned corpora
            line_count += 1

            if nfkc:
                line = unicodedata.normalize('NFKC', line)

            if protect_regex:
                line = regex.sub(protect_regex, ' ', line)

            line = '▁' + _WHITESPACE_REGEX.sub('▁', line.replace('▁', ' '))
            if inline_case:
                tokens = _NO_MIXED_CASE_REGEX.findall(line)
                tokens = [token.lower() for token in tokens]
            else:
                tokens = _SENTENCEPIECE_REGEX.findall(line)

            if delimiter is not None:
                tokens = sum((token.split(delimiter) for token in tokens), [])
                tokens = [token for token in (token.strip() for token in tokens) if token]
            if split_by_script:
                tokens = split_by_script_(tokens)

            if tokenization >= len(_TOKENIZATION_REGEXES):
                raise NotImplementedError
            else:
                # split tokens even further depending on the tokenization aggressivity level
                # 0: no split
                # 1: alphanumeric characters and others cannot mix
                # 2: letters, digits and other symbols cannot mix
                # 3: like 2, but other symbols are always on their own
                # 4: like 3, but digits are always on their own
                tokenization_regex = _TOKENIZATION_REGEXES[tokenization]
                if tokenization_regex is not None:
                    tokens = sum((tokenization_regex.findall(token) for token in tokens), [])

            for token in tokens:
                vocab[token] = vocab.get(token, 0) + 1

        return vocab, line_count
    
    @staticmethod
    def _readlines(file, buffer_size=None):
        # verbose replacement of list(islice(file, buffer_size)) which is compatible with file.tell()
        buffer = []
        while buffer_size is None or len(buffer) < buffer_size:
            line = file.readline()
            if line:
                buffer.append(line)
            else:
                break
        return buffer

    @classmethod
    def _get_vocabularies(cls, inputs, buffer_size=10000, max_lines=10**7, threads=None, verbose=False, **kwargs):
        if len(inputs) > 1:
            langs = []
            for filename in inputs:
                assert filename
                lang = regex.match(r'.+\.([a-z]+)', filename)
                assert lang is not None, 'could not infer language from file name'
                langs.append(lang.group(1))
        else:
            langs = ['any']

        fun = functools.partial(cls._get_vocabulary, **kwargs)
        pool = multiprocessing.Pool(processes=threads) if threads is None or threads > 1 else None

        # using dicts because Counter is so slow...
        # each lang has its own vocab (for oversampling)
        vocabs = defaultdict(dict)
        line_counts = defaultdict(int)

        for filename, lang in zip(inputs, langs):
            infile = sys.stdin if not filename else open(filename)

            if verbose:
                logger.info(f'reading {"STDIN" if infile is sys.stdin else filename}')
            
            vocab = defaultdict(int)
            line_count = 0

            if pool is None:
                # run in a single process
                vocab_, line_count = fun(cls._readlines(infile, max_lines))
                for k, v in vocab_.items():
                    vocab[k] += v
                continue

            read_lines = 0
            while not max_lines or read_lines < max_lines:
                buffer_size_ = min(max_lines - read_lines, buffer_size) if max_lines else buffer_size
                buffer = cls._readlines(infile, buffer_size_)
                read_lines += len(buffer)
                if not buffer:
                    break

                n = 1 + len(buffer) // (pool._processes * 8)

                result = pool.map(fun,
                    [buffer[i:i + n] for i in range(0, len(buffer), n)],
                    chunksize=8
                )

                for vocab_, line_count_ in result:
                    line_count += line_count_
                    for k, v in vocab_.items():
                        vocab[k] += v

            r = 1
            if infile is not sys.stdin:
                read_bytes = infile.tell()
                infile.seek(0, io.SEEK_END)
                total_bytes = infile.tell()
                r = total_bytes / read_bytes
            
            vocab_ = vocabs[lang]
            for k, v in vocab.items():
                vocab_[k] = vocab_.get(k, 0) + int(r * v)

            line_counts[lang] += int(r * line_count)

        return dict(vocabs), dict(line_counts)

    def _merge_vocabularies(vocabs, line_counts, temperature=1, verbose=False, **kwargs):
        if temperature is None or temperature < 0:
            temperature = 1.0        
        
        line_counts = np.array([line_counts.get(lang, 0) for lang in vocabs])

        if len(vocabs) > 1:
            probs = line_counts / line_counts.sum()
            probs = probs ** (1 / temperature)
            probs /= probs.sum()

            # oversample each lang by this multiplier (highest-resource lang's multiplier is 1)
            multipliers = ((line_counts.max() * probs) / (probs.max() * line_counts)).tolist()

            vocab = {}
            for lang, multiplier in zip(vocabs, multipliers):
                for k, v in vocabs[lang].items():
                    vocab[k] = vocab.get(k, 0) + v * multiplier

            # because of floating point multipliers, vocab counts can be non-integer values
            # round them after the sum, to limit the rounding approximation.
            vocab = {k: round(v) for k, v in vocab.items()}
        else:
            # can skip the oversampling part
            multipliers = None
            vocab = next(iter(vocabs.values()))

        if verbose:
            for i, lang in enumerate(vocabs):
                multiplier = 1 if not multipliers else round(multipliers[i], ndigits=1)
                line_count = line_counts[i]
                tokens = sum(vocabs[lang].values(), 0)
                logger.info(
                    f'[{lang}] {line_count} lines (multiplier: {multiplier}), {tokens} tokens, '
                    f'{len(vocabs[lang])} unique tokens'
                )

        return Counter(vocab)

    @staticmethod
    def _update_pair_statistics(pair, sorted_vocab, stats, indices):
        """
        Minimally update the indices and frequency of symbol pairs

        if we merge a pair of symbols, only pairs that overlap with occurrences
        of this pair are affected, and need to be updated.

        Copied from subword-nmt
        """

        # Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'
        left, right = pair
        new_pair = left + right
        
        pair_str = new_pair.replace('\\', '\\\\')
        pattern = regex.compile(r'(?<!\S)' + regex.escape(left + ' ' + right) + r'(?!\S)')

        frequencies = indices[pair]
        indices[pair] = defaultdict(int)
        stats[pair] = 0

        for j, freq in frequencies.items():
            if freq < 1:
                continue
            
            old_word, freq = sorted_vocab[j]
            word = ' '.join(old_word)
            word = pattern.sub(pair_str, word)
            word = tuple(word.split())

            sorted_vocab[j] = (word, freq)

            # find all instances of pair, and update frequency/indices around it
            i = 0
            while True:
                # find left symbol
                try:
                    i = old_word.index(left, i)
                except ValueError:
                    break
                # if left symbol is followed by right symbol, we've found an occurrence of pair (old_word[i:i+2])
                if i < len(old_word) - 1 and old_word[i + 1] == right:
                    # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                    if i:
                        prev = old_word[i - 1:i + 1]
                        stats[prev] -= freq
                        indices[prev][j] -= 1
                    if i < len(old_word) - 2:
                        # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                        # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced
                        # by the previous code block
                        if old_word[i + 2] != left or i >= len(old_word) - 3 or old_word[i + 3] != right:
                            next = old_word[i + 1:i + 3]
                            stats[next] -= freq
                            indices[next][j] -= 1
                    i += 2
                else:
                    i += 1

            i = 0
            while True:
                try:
                    i = word.index(new_pair, i)
                except ValueError:
                    break
                # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
                if i:
                    prev = word[i - 1:i + 1]
                    stats[prev] += freq
                    indices[prev][j] += 1
                # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
                # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented
                # by the previous code block
                if i < len(word) - 1 and word[i + 1] != new_pair:
                    next = word[i:i + 2]
                    stats[next] += freq
                    indices[next][j] += 1
                i += 1

    @staticmethod
    def _get_pair_statistics(vocab):
        """ Copied from subword-nmt """
        stats = defaultdict(int)
        indices = defaultdict(lambda: defaultdict(int))

        for i, (word, freq) in enumerate(vocab):
            for prev_char, char in zip(word, word[1:]):
                stats[prev_char, char] += freq
                indices[prev_char, char][i] += 1

        return stats, indices

    @staticmethod
    def _prune_stats(stats, big_stats, threshold):
        """
        Prune statistics dict for efficiency of max()

        The frequency of a symbol pair never increases, so pruning is generally safe
        (until the most frequent pair is less frequent than a pair we previously pruned)
        big_stats keeps full statistics for when we need to access pruned items

        Copied from subword-nmt
        """
        for item, freq in list(stats.items()):
            if freq < threshold:
                del stats[item]
                if freq < 0:
                    big_stats[item] += freq
                else:
                    big_stats[item] = freq

    @staticmethod
    def detokenize(tokens: list[str]) -> str:
        return detokenize(tokens)

    def detokenize_on_the_fly(self, tokens: Iterable[str]) -> Iterator[tuple[str, list[str]]]:
        prev_tokens = []
        for token in tokens:
            if not token:
                continue
            if prev_tokens and token[0] == '▁':
                yield detokenize(prev_tokens, strip=False), prev_tokens
                prev_tokens = []
            prev_tokens.append(token)
        if prev_tokens:
            yield detokenize(prev_tokens, strip=False), prev_tokens

    @staticmethod
    def build_dict(
        vocab, dict_path=None, dict_custom_symbols=[], dict_placeholders=0, dict_padding_offset=4,
        dict_padding_factor=8, dict_min_freq=10, dict_max_size=None, **_,
    ):
        """
        Used to create a Pasero dictionary from a list of tokens.
        """
        dictionary = dict.fromkeys(['<T>', '<U>', '<BT>', '<PHL>', mask], 0)
        
        if not isinstance(vocab, dict):  # vocab can be a list or set
            vocab = dict.fromkeys(vocab, 0)

        vocab = dict(vocab)    # convert Counter to dict (because Counter's update has a different behavior)
        # count all characters and add missing characters to the dictionary
        chars = defaultdict(int)
        for word, count in vocab.items():
            if word not in dictionary:   # do not spell out special tokens
                for char in word:
                    chars[char] += count
        vocab.update(chars)   # unexpected behavior with Counter
        vocab = {w: c for w, c in vocab.items() if not c or c >= dict_min_freq}
        vocab = sorted(vocab.items(), key=lambda p: (-p[1], p[0]))  # sort by count, then alphabetically
        dictionary.update(dict(vocab))
        
        special_symbols = []
        for token in sorted(dict_custom_symbols):
            if token not in dictionary:
                special_symbols.append((token, 0))

        i = 0
        for _ in range(dict_placeholders):
            special_symbols.append((f'madeupword{i:04}', 0))
            i += 1

        dictionary = list(dictionary.items())

        if dict_max_size is not None:
            assert len(special_symbols) < dict_max_size
            dictionary = dictionary[:dict_max_size - len(special_symbols)]
        
        dictionary += special_symbols

        while (len(dictionary) + dict_padding_offset) % dict_padding_factor != 0:
            dictionary.append((f'madeupword{i:04}', 0))
            i += 1

        if dict_path is not None:
            if dict_path == '-':
                dict_file = sys.stdout
            else:
                dirname = os.path.dirname(dict_path)
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
                dict_file = open(dict_path, 'w')
            dict_file.writelines(f'{token} {count}\n' for token, count in dictionary)
        return dictionary
