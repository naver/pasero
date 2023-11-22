# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import json
import sys
import os
import copy
import yaml
import random
from typing import Optional, Union, Any, get_args, get_origin
from pasero import evaluation


class argument:
    def __init__(
        self,
        default: Optional[Any] = None,
        defaults: Optional[dict[str, Any]] = None,
        help: Optional[str] = None,
        aliases: list[str] = [],
        choices: Optional[list[Any]] = None,
        nargs: Union[str, int, None] = None,
        positional: bool = False,
    ):
        if defaults:
            assert default is None, "arguments with a 'defaults' property cannot have 'default'"
        self.default = default
        self.defaults = defaults
        self.help = help
        self.aliases = aliases
        self.choices = choices
        self.nargs = nargs
        self.positional = positional
        self.api_opt = not positional


def is_optional(type_):
    """ Returns True for Optional[T] types """
    types = get_args(type_)
    return get_origin(type_) is Union and len(types) == 2 and type(None) in types

def is_list(type_):
    """ Returns True for list[T] types """
    return get_origin(type_) is list

def optional_type(type_):
    """ Optional[T] -> T """
    return next(t for t in get_args(type_) if t is not type(None))

def list_type(type_):
    """ list[T] -> T """
    return get_args(type_)[0]
    
def union_types(type_):
    """
    Union[X, Y] -> (X, Y)
    T -> T
    """
    if get_origin(type_) is Union:
        return tuple(t for t in get_args(type_) if t is not type(None))
    else:
        return (type_,)


class Config:
    @classmethod
    def arguments(cls) -> list[argument]:
        """
        Get the list of arguments for this task. Uses the type annotations to specify the argument's `type` and 
        `name` attributes. The default values, help message and other fields are specified through the `argument`
        class constructor.
        Note that subclasses may override a superclass argument, but only by defining a class attribute with the same
        name (which will alter the original argument's type and default value).
        """
        arguments = {}

        for cls in reversed(cls.__mro__[:-1]):  # loop over all super classes (excluding `object`)
            # reversed to go from super classes to sub classes (Config -> ... -> cls)
            
            annotations = getattr(cls, '__annotations__', {})
            for name, type in annotations.items():
                arg = copy.copy(getattr(cls, name))  # we don't want to accidentally modify the default values
                # of superclass arguments

                if isinstance(arg, argument):
                    arg.type = type
                    arg.name = name
                    if arg.defaults:
                        assert is_optional(arg.type), 'arguments with a "defaults" field should have the `Optional` ' \
                            'type'
                    arguments[name] = arg
                elif name in arguments:  # subclasses are allowed to override the default value of an argument by 
                    # specifying a class attribute (not an argument) with the same name (see TransformerModel's
                    # subclasses)
                    arguments[name].type = type
                    arguments[name].default = arg
                    arguments[name].defaults = None

        return list(arguments.values())

    def set_defaults(self, task: str) -> None:
        """
        Some default values may depend on the task and are specified as a dict field named "defaults".
        
        This method is called by `TrainingConfig` and `DecodingAPIConfig` after parsing the user options and YAML files 
        and once the task is known.

        This is different to the "default" field, which is automatically used to initialize its attributes when
        initializing a `Config`.

        Note that the use of "defaults" requires the argument to be Optional, because its value might be temporarily 
        None (in which case, the task's default will be used). This does not mean that this argument's value is 
        actually allowed to be None.
        
        This field also exclusive with "default" (which would prevent the argument's value from ever being None)
        """
        parent_tasks = get_task_config_cls(task).mro()  # when specifying a default value for 'translation' task, 
        # we want it to apply to its subclasses (e.g., 'speech_translation'), unless a default value is also 
        # specified for those subclasses.

        def find_closest_task(tasks: list[str]) -> Optional[str]:
            # try to find this task or one of its parents in given task list
            tasks = {get_task_config_cls(task): task for task in tasks}
            for parent_task in parent_tasks:
                if parent_task in tasks:
                    return tasks[parent_task]

        for arg in self.arguments():
            value = getattr(self, arg.name)
            if value is None and arg.defaults:
                task = find_closest_task(arg.defaults)
                if task is not None:
                    setattr(self, arg.name, arg.defaults[task])

    def __init__(self, *opts: Union[str, 'Config'], strict: bool = True, **kwargs):
        """
        Can be initialized in 3 different ways:
        - with a list of command line options
        - with another configuration (which can be a subclass of this one)
        - keyword arguments

        For instance:
        ```
        # TrainingConfig is a superclass of EvalConfig, but we may need to retrieve only the subset of eval options:
        cfg = TrainingConfig(...)
        eval_cfg = EvalConfig(cfg)

        # An EvalConfig instance can also be created from command line options:
        cfg = EvalConfig(['--metrics', 'bleu', '--eval-lc'])
        cfg.as_dict()
        {
            'metrics': ['bleu'],
            'eval_lc': True,
            ...
        }

        # which is equivalent to:
        cfg = EvalConfig(metrics=['bleu'], eval_lc=True)
        ```
        """
        super().__init__()

        for arg in self.arguments():
            setattr(self, arg.name, arg.default)

        if opts and isinstance(opts[0], Config):
            assert len(opts) == 1 and not kwargs
            kwargs = opts[0].as_dict()
            strict = False
            opts = []

        self.parse_dict(kwargs, strict=strict)
        self.parse_args(opts, strict=strict)

    def as_dict(self) -> dict:
        """ Convert this configuration into a Python dictionary """
        cfg = {
            arg.name: getattr(self, arg.name)
            for arg in self.arguments()
        }
        return dict(sorted(cfg.items()))

    def parse_dict(self, config: dict[str, Any], strict: bool = False):
        """
        Initialize this config's attributes with given dictionary, whose keys correspond to argument names.
        Also checks that the given values have the right type for the corresponding arguments.
        If `strict` is False, unknown options are allowed and are returned by this method (which will allow to parse 
        them with another config)
        """
        unknown_config = {}
        
        for name, value in config.items():
            if name == 'continue':
                name = 'continue_'

            if hasattr(self, name):
                self._check_type(name, value)
                setattr(self, name, value)
            else:
                unknown_config[name] = value
        
        if unknown_config and strict:
            raise AttributeError("unknown option(s): " + ', '.join(unknown_config))
        
        return unknown_config

    def _get_parser(self, add_help: bool = False) -> argparse.ArgumentParser:
        """
        Create an `argparse` parser to parse command-line arguments (with `parse_args`) and initialize this config's 
        attributes with those.
        """
        parser = argparse.ArgumentParser(add_help=add_help, allow_abbrev=False)
        
        for arg in self.arguments():
            opts = {}

            if arg.positional:
                name = arg.name.strip('_')
                assert not arg.aliases
            else:
                name = '--' + arg.name.strip('_').replace('_', '-')
                opts['dest'] = arg.name

            type_ = arg.type
            if is_optional(type_):
                type_ = optional_type(type_)

            if arg.nargs is not None:
                opts['nargs'] = arg.nargs

            if is_list(type_):
                type_ = list_type(type_)
                opts.setdefault('nargs', '*')

            types = union_types(type_)
            type_ = next(  # in case of multiple types, we take the first one which is compatible with argparse
                (t for t in types if t in (bool, int, float, str)),
                types[0]
            )

            if type_ not in (bool, int, float, str, dict):
                raise NotImplementedError

            if type_ is dict:  # only for YAML config
                continue

            if type_ is bool:
                opts['action'] = argparse.BooleanOptionalAction

            parser.add_argument(
                name,
                *arg.aliases,
                type=type_,
                default=getattr(self, arg.name),
                help=arg.help,
                choices=arg.choices,
                **opts,
                # TODO: required options
            )
        
        return parser

    def parse_args(self, opts: Optional[list[str]] = None, strict: bool = False, add_help: bool = False):
        """
        Parse given command-line options or the main program's options (`sys.argv[1:]`) if none are given 
        and initialize this config's attributes with the parsed options.
        For instance, the command-line arguments `--beam-size 5` will set the attribute named `beam_size` to 5.  
        
        If `strict` is False, unknown options are allowed and are returned by this method (which will allow to parse 
        them with another config)
        """
        parser = self._get_parser(add_help=add_help)
        namespace, other_opts = parser.parse_known_args(opts)

        for name, value in namespace.__dict__.items():
            setattr(self, name, value)
        
        if other_opts and strict:
            raise AttributeError("unknown option(s): " + ' '.join(other_opts))

        return other_opts

    def _check_type(self, name, value):
        """
        Check that given `value` has the correct type for argument `name`
        """
        arg = next(arg for arg in self.arguments() if arg.name == name and arg.api_opt)
        type_ = arg.type
        
        if is_optional(type_):
            if value is None:
                return
            type_ = optional_type(type_)
        else:
            assert value is not None, f"option '{name}' cannot be None"

        def check_base_type(value, types):
            for type_ in types:
                assert type_ in (bool, int, float, str, dict), f"type '{type_.__name__} is not supported"
            type_names = '/'.join(t.__name__ for t in types)
            assert isinstance(value, types) or type_ is float and isinstance(value, int), \
                f"option '{name}' expects values of type '{type_names}', got '{type(value).__name__}'"

        if is_list(type_):
            assert isinstance(value, list), f"option '{name}' expects a list, got '{type(value).__name__}'"

            nargs = arg.nargs or '*'
            choices = arg.choices

            if isinstance(nargs, int):
                assert len(value) == nargs, f"option '{name}' excepts {nargs} values"
            elif nargs == '+':
                assert len(value) > 0, f"option '{name}' excepts at least 1 value"
            elif nargs != '*':
                raise NotImplementedError

            type_ = list_type(type_)
            types = union_types(type_)
            for value_ in value:
                check_base_type(value_, types)
                assert not choices or value_ in choices, f"{repr(value_)} is not a valid choice for option '{name}'"
        else:
            types = union_types(type_)
            check_base_type(value, types)

    @classmethod
    def parse_str(cls, name: str, value: str) -> Union[int, float, str, bool]:
        """
        Parse given `value` string and convert into the correct type for argument `name`. Raise an error if it does 
        not parse into this type. For instance:

        `parse_str('beam_size', '5')` => `5`

        `parse_str('beam_size', 'banana') => ValueError
        """
        for arg_name, arg_type in cls.__annotations__.items():
            if arg_name != name:
                continue

            trues = ('1', 'true', 'yes', 'on')
            falses = ('0', 'false', 'no', 'off')

            if arg_type in (int, float, str):
                return arg_type(value)
            elif arg_type is bool and value.lower() in trues:
                return True
            elif arg_type is bool and value.lower() in falses:
                return False
            elif arg_type is bool:
                raise ValueError(f"invalid value '{value}' for argument type '{arg_type.__name__}'")
            else:
                raise ValueError(f"unsupported argument type '{arg_type.__name__}'")

        raise ValueError(f"unknown argument '{name}'")


def parse_help(*configs: list[Config]):
    if '-h' in sys.argv or '--help' in sys.argv:
        parsers = [
            cfg.get_parser(add_help=False) for cfg in configs
        ]
        parser = argparse.ArgumentParser(parents=parsers, conflict_handler='resolve')
        parser.parse_args(['-h'])


class DistributedConfig(Config):
    dp_size: Optional[int] = argument(
        help='number of GPUs used for training or decoding with data parallelism',
    )
    tp_size: int = argument(
        default=1,
        help='apply tensor parallelism with this size (i.e., Transformer layers will be sharded across this many GPUs)',
    )
    start_rank: int = argument(
        default=0,
        help='the starting rank of this node (e.g., for 2x4 training, this would be 0 on the master node and 4 on the '
             'second node)'
    )
    distributed_init_method: Optional[str] = argument(
        help="method to initiate communication between nodes in multi-node training or decoding. This can be either "
             "via TCP: 'tcp://MASTER_ADDR:SOME_PORT', or via a file on a shared filesystem: "
             "'file://${HOME}/tmp/SOME_FILE'"
    )
    dtype: str = argument(
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help='which data type to use: float16/bfloat16 for mixed-precision training and half-precision decoding, and '
             'float32 for full-precision (default: float16)'
    )
    seed: Optional[int] = argument(
        help='seed used to initialize the random number generators (default: random seed). Note that the training data '
             'loading pipeline is non-deterministic'
    )
    sequence_parallel: bool = argument(
        default=True,
        help='apply sequence parallelism in addition to tensor parallelism (i.e., data will be sharded along the batch '
             'dimension. This will be disabled at inference'
    )
    
    # The parameters below shouldn't be set by the user, but automatically by `utils.setup_distributed`
    dp_rank: int = 0
    tp_rank: int = 0
    dp_local_rank: int = 0
    dp_local_size: int = 1
    
    @property
    def distributed_world_size(self):
        return (self.dp_size or 1) * (self.tp_size or 1)

    @property
    def distributed_rank(self):
        return self.dp_rank * self.tp_size + self.tp_rank


class TrackerConfig(Config):
    tracker: str = argument(
        default='none',
        choices=['wandb', 'neptune', 'mlflow', 'none'],
        help='which experiment tracker to use (default: none)'
    )
    tracker_project_name: Optional[str] = argument(
        help='name of the project this experiment should belong to in the experiment tracker'
    )
    tracker_run_name: Optional[str] = argument(
        help='name of this training run in the experiment tracker'
    )


class DecodingConfig(Config):
    max_output_len: int = argument(
        default=100,
        help='maximum number of new new tokens to be generated (does not count the length of the prompt)'
    )
    min_output_len: int = argument(
        default=0,
        help='minimum number of new new tokens to be generated (does not count the length of the prompt)'
    )
    beam_size: Optional[int] = argument(
        defaults={
            'language_modeling': 1,
            'translation': 5,
        },
        help='beam size used during decoding (set to 1 for greedy generation)'
    )
    repeat_penalty: float = argument(
        default=1.0,
        help='penalize repeated tokens by this amount (1 means no penalty), not supported in beam search'
    )
    sampling: bool = argument(
        default=False,
        help='use sampling instead of beam search for decoding'
    )
    sampling_topk: int = argument(
        default=0,
        help='sample from the k best tokens'
    )
    sampling_topp: float = argument(
        default=1,
        help='use nucleus sampling with this probability'
    )
    sampling_temperature: float = argument(
        default=1.0,
        help='softmax temperature when sampling (<1: closer to greedy decoding, >1: closer to uniform sampling)'
    )
    len_penalty: float = argument(
        default=1.0,
        help='normalize the scores of the n-best beam search hypotheses by their length to this power'
    )



class EvalConfig(Config):
    teacher_forcing: bool = argument(
        default=False,
        help='force the model to generate the reference'
    )
    bleu_tok: Optional[str] = argument(
        aliases=['--bleu-tokenize'],
        choices=evaluation.BLEU_TOKENIZERS,
        help="Tokenization method to use for BLEU. If not provided, defaults to 'zh' for Chinese, 'ja-mecab' for "
             "Japanese and '13a' (mteval) otherwise."
    )
    eval_lc: bool = argument(
        aliases=['--bleu-lc'],
        default=False,
        help='perform case insensitive BLEU evaluation'
    )
    metrics: Optional[list[str]] = argument(
        choices=evaluation.METRICS,
        defaults={
            'language_modeling': [],
            'translation': ['chrf', 'bleu', 'chrf++', 'spbleu', 'len_ratio'],
        },
        help='evaluation metrics to compute'
    )
    merge_bpe: bool = argument(
        default=False,
        help="used in conjunction with '--tokenizer none' when the data is pre-tokenized: detokenizes both the "
             "hypotheses and references before BLEU evaluation"
    )


class NoiseConfig(Config):
    space_noise: float = argument(
        default=0.0,
        help='drop or insert whitespaces with this probability'
    )
    punct_noise: float = argument(
        default=0.0,
        help='drop punctuation symbols with this probability'
    )
    char_noise: float = argument(
        default=0.0,
        help='apply character-level operations with this probability'
    )
    noise_ops: list[str] = argument(
        choices=['ins', 'del', 'sub', 'swap'],
        default=['ins', 'del', 'sub', 'swap'],
        nargs='+',
        help='list of allowed character noise operations (insertions, deletions, substitutions or swaps), operations '
             'are sampled uniformly from this list'
    )
    word_noise: float = argument(
        default=0.0,
        help='drop entire words with this probability'
    )
    masking: float = argument(
        default=0.0,
        help='mask entire words with this probability'
    )


class PreprocessingConfig(NoiseConfig):
    tokenizer: str = argument(
        choices=['pasero', 'sentencepiece', 'none', 'hf', 'char'],
        default='pasero',
        help="BPE implementation to use. Set to 'none' to disable BPE tokenization"
    )
    tokenizer_path: Optional[str] = argument(
        help="path to the BPE model, absolute or relative to DATA_DIR (at training) or MODEL_DIR (at inference)"
    )
    tokenizer_vocab: Optional[str] = argument(
        help="path to the vocabulary countaining BPE token frequencies used for threshold-based filtering (if any)"
    )
    hf_add_prefix_space: bool = argument(
        default=False,
        help='the BLOOM tokenizer treats the first word in the sentence differently, get around this behavior by '
             'prefixing each sentence with a whitespace'
    )
    vocabulary_threshold: Optional[int] = argument(
        help='prevent any BPE token whose frequency is lower than this threshold from being generated (frequencies are '
             'obtained from the vocabulary file specified with --tokenizer-vocab). If not set to zero or not set and '
             '--tokenizer-vocab VOCAB is set, only subwords appearing in VOCAB will be allowed'
    )
    inline_case: bool = argument(
        default=False,
        help='apply inline casing: put all text to lowercase and add special tokens indicating the case '
             'of the preceding subword (inline casing is activated by default with --tokenizer pasero)'
    )
    dict: str = argument(
        aliases=['--source-dict'],
        default='dict.txt',
        help="path to the source dictionary, absolute or relative to DATA_DIR (at training) or MODEL_DIR (at inference)"
    )
    bpe_dropout: float = argument(
        default=0.0,
        help='apply BPE dropout to the source training data with this rate (default: disabled)'
    )
    spell_out: float = argument(
        default=0.0,
        help='spell out training source words with this probability (default: disabled)'
    )
    keep_whitespaces: Optional[bool] = argument(
        defaults={
            'language_modeling': True,
            'translation': False,
        },
        help='do not remove or normalize whitespaces or non-printing characters'
    )
    normalize_punctuation: bool = argument(
        default=False,
        help='normalize punctuation with rules from the Stopes library'
    )
    blacklist: list[str] = argument(
        default=[],
        help='list of tokens that should not be generated'
    )
    stop_sequences: list[str] = argument(
        default=[],
        help="list of whitespace-delimited token sequences that should stop generation (in addition to '</s>'), not "
             "supported in beam search"
    )
    strip_prompt: bool = argument(
        default=True,
        help='remove the prompt from the detokenized output'
    )


class TaskConfig(PreprocessingConfig):
    batch_size: int = argument(
        default=4096,
        help='maximum number of tokens in a batch the tokens in a pair of lines is counted as the maximum of source '
             'and target length'
    )
    batch_size_multiple: Optional[int] = argument(
        help='force the batch sizes to be a multiple of this value. This can impact performance with mixed precision '
             'training'
    )
    lines_per_batch: Optional[int] = argument(
        help='maximum number of lines in a batch'
    )

    domains: Optional[list[str]] = argument(
        aliases=['--domain'],
        nargs='+',
        help='domain(s) used for defining domain tags and domain-specific parameters)'
    )
    def replace_placeholders(self, path: str) -> str: raise NotImplementedError

    def finalize(self):
        for opt in 'tokenizer_path', 'tokenizer_vocab', 'dict':
            value = getattr(self, opt, None)
            if value:
                setattr(self, opt, self.replace_placeholders(value))

    def set_max_length(self, model_cfg: 'TransformerConfig'):
        raise NotImplementedError


class DecodingAPIConfig(DistributedConfig, DecodingConfig):
    dp_size: int = 1
    task_cfg: Optional[TaskConfig] = None
    
    task: str = argument(
        choices=[
            'translation',
            'speech_translation',
            'language_modeling',
            'doc_level_translation',
            'nllb_translation',
            'dialogue',
        ],
        default='translation',
    )
    model: str = argument(
        help='path to a model directory or checkpoint'
    )
    model_dir: Optional[str] = argument(
        help='path to a model directory (prefer using --model)'
    )
    ckpt: Optional[str] = argument(
        help='path to a model checkpoint (prefer using --model)'
    )
    other_ckpt: list[str] = argument(
        default=[],
        help='paths to other checkpoints that will be merged with MODEL'
    )
    ensemble_ckpt: list[str] = argument(
        default=[],
        help='list of checkpoints to use in an ensemble with MODEL'
    )
    moe_stats: bool = argument(
        default=False,
        help='output gate statistics for Mixtures of Experts after decoding corpora'
    )
    devices: Optional[list[str]] = argument(
        aliases=['--device'],
        help='devices on which to distribute the encoder and decoder layers (pipeline parallelism)'
    )
    benchmark: bool = argument(
        default=False,
        help='compute the time spent in each Transformer component (this will slow down decoding)'
    )
    flexible: bool = argument(
        default=False,
        help='load the checkpoint anyway if it has missing or unexpected parameters'
    )
    model_args: Optional[str] = argument(
        help='json dictionary as a string defining model arguments that should be overloaded'
    )
    encoder_adapters: Optional[list[str]] = argument(
        help="names of the adapters that should be stacked at each encoder layer (only for the adapter_transformer "
             "arch). This overrides the arguments stored in the model checkpoint. Set to empty to disable encoder "
             "adapters. By default, if the model was trained with '--encoder-adapters-by' and '--decoder-adapters-by', "
             "adapters will be activated at decoding based on the given languages or domains"
    )
    decoder_adapters: Optional[list[str]] = argument(
        help="names of the adapters that should be stacked at each decoder layer (only for the adapter_transformer "
             "arch). This overrides the arguments stored in the model checkpoint. Set to empty to disable encoder "
             "adapters"
    )
    encoder_decoder_swapping: bool = argument(
        default=False,
        help='save GPU memory by moving the encoder and decoder to CPU when they are not needed'
    )
    arch: Optional[str] = argument(
        help='override the model architecture'
    )

    def __init__(self, *opts, strict: bool = True, **kwargs):
        """
        Decoding options can be specified in 3 different ways (from lowest to highest precedence):

        - YAML config file ('inference.yaml' in the model directory)
        - keyword arguments (given via `kwargs`)
        - command line arguments (given via `opts`)
        """
        super().__init__(*opts, strict=False, **kwargs)
        
        # set 'model_dir' and 'ckpt' according to 'model' (which can be either a checkpoint path or model directory)
        if self.model is not None:
            if os.path.isfile(self.model):
                self.ckpt = self.model
            else:
                self.model_dir = self.model

        if self.ckpt is not None and self.model_dir is None:
            self.model_dir, self.ckpt = os.path.split(self.ckpt)

        assert not self.model_dir or os.path.isdir(self.model_dir), 'given model directory does not exist'

        yaml_path = os.path.join(self.model_dir, 'inference.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path) as yaml_file:
                yaml_opts = yaml.safe_load(yaml_file)
                for k, v in self.parse_dict(yaml_opts).items():
                    kwargs.setdefault(k, v)

        # set default options that depend on the task
        self.set_defaults(self.task)

        # these options have higher precedence over 'inference.yaml'
        kwargs = self.parse_dict(kwargs, strict=False)
        cli_opts = self.parse_args(opts, strict=False)
        task_cfg = get_task_config(self.task)
        parse_help(self, task_cfg)
        kwargs = task_cfg.parse_dict(kwargs, strict=False)
        cli_opts = task_cfg.parse_args(cli_opts, strict=strict)  # all command-line options should be accounted for

        self.task_cfg = task_cfg
        task_cfg.finalize()
        
        if self.ckpt is None:
            for ckpt_name in 'model_best.bin', 'model_latest.bin', 'model_last.bin':
                ckpt_path = os.path.join(self.model_dir, ckpt_name)
                if os.path.isfile(ckpt_path):
                    self.ckpt = ckpt_path
                    break
            assert self.ckpt is not None, f"no checkpoint found in '{self.model_dir}'"
        else:
            self.ckpt = os.path.join(self.model_dir, self.ckpt)

        assert os.path.isfile(self.ckpt), f"checkpoint '{self.ckpt}' does not exist"
        assert self.min_output_len <= self.max_output_len
        assert (self.beam_size == 1 or self.sampling) or not self.task_cfg.stop_sequences, ('beam search does not '
            'support --stop-sequences')

        if self.seed == -1:
            # pick the seed at random
            self.seed = random.randrange(10**9)
        elif self.seed is None:
            # default behavior is deterministic
            self.seed = 42

    def as_dict(self) -> dict:
        dict_ = super().as_dict()
        if self.task_cfg is not None:
            dict_.update(self.task_cfg.as_dict())
        return dict(sorted(dict_.items()))


class DecodingCLIConfig(DecodingAPIConfig, EvalConfig):
    model: str = argument(
        positional=True,
        help='path to a model directory or checkpoint'
    )
    input: Optional[list[str]] = argument(
        aliases=['-i'],
        nargs='+',
        help='input file(s) (default: standard input)'
    )
    max_lines: Optional[int] = argument(
        help='read at most this many lines from input'
    )
    output: Optional[list[str]] = argument(
        aliases=['-o'],
        nargs='+',
        help='where to write the decoding outputs (default: standard output), can contain placeholders (e.g., {pair}, '
             '{src}, {tgt}, {lang}'
    )
    reference: Optional[list[str]] = argument(
        aliases=['-r'],
        nargs='+',
        help='file containing the references against which evaluation metrics should be computed. If it is specified, '
             'any (INPUT, REFERENCE) line pair where either side is empty is skipped (to deal with multi-aligned '
             'corpora like TED, where absent translations in a language are specified with an empty line)'
    )
    eval_corpus: Optional[str] = argument(
        aliases=['-e'],
        help='evaluate this corpus (infers the input and ref paths from -s and -t), can contain placeholders '
             '(e.g., {pair}, {src}, {tgt}, {lang}'
    )
    continue_: bool = argument(
        default=False,
        help='if the output file already exists, resume decoding where we left off: count the lines in it and skip '
             'this many lines in INPUT and REFERENCE'
    )
    buffer_size: int = argument(
        aliases=['-n'],
        default=100,
        help='number of lines to read from INPUT at once before building batches. Larger values can result in faster '
             'decoding as batches might bigger and more efficiently sorted and padded. Set to 1 for interactive '
             'line-by-line decoding. Reduce if OOM errors occur (other levers are --batch-size and --max-output-len)'
    )
    verbose: bool = argument(
        aliases=['-v'],
        default=False,
        help='show more information when decoding (scores, tokenization, etc.)'
    )
    quiet: bool = argument(
        aliases=['-q'],
        default=False,
        help='no decoding output is displayed to standard output'
    )
    log_file: Optional[str] = argument(
        help='path for the logging output relative to MODEL_DIR'
    )

    def __init__(self, *opts, strict: bool = True, **kwargs):
        opts = opts or sys.argv[1:]
        # FIXME: will do an error if "model" is not given, even with "-h" (parse_help is called in DecodingAPIConfig
        # after we setup the task configuration)
        super().__init__(*opts, strict=strict, **kwargs)


class TrainingDatasetConfig(Config):
    buffer_size: int = argument(
        default=100000,
        help='number of line pairs to load and pre-process before batching. Larger values will take more memory and '
             'slow down the beginning of training, but will result in more efficient batching (because the buffer is '
             'sorted by length before padding and batching). However, the buffer size should remain smaller than the '
             'training data size, to avoid having duplicates in the batches'
    )
    batch_by: Optional[list[str]] = argument(
        choices=['lang', 'source_lang', 'target_lang', 'domain'],
        help="homogeneous batching based on this metadata. For example, '--batch-by source_lang target_lang' will build "
             "batches that only contain data of one language pair. This is important when training language-specific "
             "parameters (e.g., with --encoder-adapter-by). This will cause batches to be less packed, which can be "
             "mitigated by setting a higher --buffer-size"
    )
    lang_temperature: float = argument(
        default=1.0,
        help="compute the sampling probabilities per language pair using this temperature parameter. "
             "1 (default): the sampling probability of a language pair is proportional to its data size. "
             ">1: closer to uniform sampling, lower-resource languages have a higher chance of being sampled"
    )
    dataloader_workers: int = argument(
        default=1,
        help="number of processes in the PyTorch DataLoader, used for converting data to tensors and padding. "
             "A value of 0 will result in synchronous processing. Note that each GPU has its own DataLoader with this "
             "many processes. Tip: set this value to 0, and '--dataset-type simple' to be allowed to put breakpoints "
             " in the data loading code"
    )
    reset_line_index: bool = argument(
        default=False,
        help='reset the file containing containing the line positions (e.g., if it is broken)'
    )
    cache_line_index: bool = argument(
        default=True,
        help='cache the line positions on the disk for future runs (an index of the training corpora is built at the '
             'beginning of training to count the lines, compute sampling probabilities, and speed up random reads; '
             'this process is very slow as it requires to read all the files)'
    )
    line_index_path: Optional[str] = argument(
        help='path to the file containing the cached line positions'
    )
    block_size: int = argument(
        default=256,
        help='to increase reading speed, consecutive lines from a given corpus are grouped in blocks of this size '
             '(sampling and line indexing are done on these blocks)'
    )
    num_workers: int = argument(
        default=4,
        help='number of processes used for tokenizing the training data'
    )
    shuffle: bool = argument(
        default=True,
        help='shuffle the corpus lines and batches'
    )
    max_lines: Optional[int] = argument(
        help='maximum number of line pairs to load per training corpus'
    )
    dataloader_pin_memory: bool = argument(
        default=True,
        help='do memory pinning in Pytorch Dataloader'
    )
    dataloader_prefetch_factor: int = argument(
        default=2,
        help='prefetch factor in Pytorch DataLoader'
    )
    truncate: bool = argument(
        default=False,
        help='truncate training examples that are too long rather than skipping them'
    )
    close_files: bool = argument(
        default=False,
        help="close files in between reads (to avoid 'too many open files' errors)"
    )
    store_files_under: int = argument(
        default=2**27,
        help='store files under this size in memory (default: 128 MiB)'
    )


class DynamicTrainingDatasetConfig(TrainingDatasetConfig):
    per_gpu_batching: bool = argument(
        default=False,
        help='each GPU process handles its own reading and batching, instead of having the master send batches to all '
             'processes via a queue'
    )


class SimpleDynamicTrainingDatasetConfig(DynamicTrainingDatasetConfig):
    # values that work well with small speech datasets:
    buffer_size: int = 5000
    dataloader_workers: int = 4
    dataloader_prefetch_factor: int = 10
    store_files_under: int = 0  # TODO: benchmark store_files_under > 0. This will cause all small files to be
    # stored in memory in each of the workers (DATALOADER_WORKERS * NUM_GPUS times), which could lead to "
    # excessive memory consumption, especially when combined with --cache-data. An alternative to this option "
    # to avoid opening too many files at the same time is --close-files
    cache_data: bool = argument(
        default=False,
        help='save the pre-processed data into memory to speed up the next epochs'
    )
    max_cache_size: int = argument(
        default=128,
        help='maximum size of the cache per node in GiB'
    )


class DebugTrainingDatasetConfig(SimpleDynamicTrainingDatasetConfig):
    buffer_size: int = 1000
    dataloader_workers: int = 0
    dataloader_prefetch_factor: int = 1


MODEL_CONFIGS = {}

def register_model_config(*names: str):
    assert len(names) >= 1
    def wrapper(cls):
        cls.name = names[0]
        for name in names:
            MODEL_CONFIGS[name] = cls
        return cls
    return wrapper


@register_model_config('transformer')
class TransformerConfig(Config):
    _arch: str = 'transformer'

    encoder_layers: int = argument(
        default=6,
        help='number of Transformer layers in the encoder'
    )
    decoder_layers: int = argument(
        default=6,
        help='number of Transformer layers in the decoder'
    )
    shared_embeddings: bool = argument(
        default=True,
        help='share encoder and decoder embeddings'
    )
    conv_kernel_sizes: Optional[list[int]] = argument(
        nargs='*',
        help='apply 1D convolutions to the input sequence with these kernel sizes'
    )
    conv_strides: Optional[list[int]] = argument(
        nargs='*',
        help='1D convolution layers will have these strides (default: 2)'
    )
    conv_activation: Optional[str] = argument(
        default='glu',
        choices=['glu', 'gelu'],
        help='activation function to use at the end of each convolution layer',
    )
    input_dim: Optional[int] = argument(
        help='dimension of the binary inputs'
    )
    conv_input_dim: Optional[int] = argument(
        help='dimension of the convolution input: if this is different than --input-dim a linear projection followed '
             'by a ReLU is applied'
    )
    conv_channels: Optional[int] = argument(
        help='inner dimension of the convolution layers (default: same as --conv-input-dim)'
    )
    embed_dim: int = argument(
        default=512,
        help='dimension of the embeddings and Transformer layer outputs'
    )
    encoder_ffn_dim: int = argument(
        default=2048,
        help='inner dimension of the encoder feed-forward blocks'
    )
    decoder_ffn_dim: int = argument(
        default=2048,
        help='inner dimension of the decoder feed-forward blocks'
    )
    encoder_attention_heads: int = argument(
        default=8,
        help='number of encoder self-attention heads'
    )
    decoder_attention_heads: int = argument(
        default=8,
        help='number of decoder self-attention and cross-attention heads'
    )
    attention_heads_kv: Optional[int] = argument(
        help='number of attention heads for the keys and values, if different than for the queries (1 for multi-query '
             'attention'
    )
    sliding_window: Optional[int] = argument(
        help='use self-attention with a sliding window of this size'
    )
    scale_attn: bool = argument(
        default=True,
        help='divide the attention query-key dot product by the square root of the head dimension'
    )
    check_inf: bool = argument(
        default=False,
        help='replace infinite values from the outputs of each transform block with large values (technique used to '
             'train and run T5 in float16)'
    )
    attention_key_bias: bool = argument(
        default=True,
        help='whether the attention key projections have a bias parameter'
    )
    dropout: float = argument(
        default=0.1,
        help='amount of dropout applied to the Transformer'
    )
    decoder_dropout: Optional[float] = argument(
        help='amount of dropout applied to the Transformer decoder (default: same as --dropout)'
    )
    attention_dropout: float = argument(
        default=0.0,
        help='amount of dropout applied to the attention weights'
    )
    activation_dropout: float = argument(
        default=0.0,
        help='amount of dropout applied inside the feed-forward blocks'
    )
    label_smoothing: Optional[float] = argument(
        defaults={
            'language_modeling': 0.0,
            'translation': 0.1,
        },
        help="amount of label smoothing. Regularization technique which assigns this probability mass to the 'wrong' "
             "labels"
    )
    tied_output_projection: bool = argument(
        default=True,
        help='tie the decoder embeddings with the output projection matrix'
    )
    activation_fn: str = argument(
        choices=['relu', 'gelu', 'gelu_tanh', 'swiglu'],
        default='relu',
        help='activation function used in the feed-forward blocks'
    )
    has_bias: bool = argument(
        default=True,
        help='whether the linear layers should have a bias or not (e.g., Llama does not have any bias)'
    )
    encoder_prenorm: bool = argument(
        default=False,
        help='put the layer normalization before each Transformer encoder block rather than after'
    )
    decoder_prenorm: Optional[bool] = argument(
        default=False,
        help='put the layer normalization before each Transformer decoder block rather than after'
    )
    encoder_embed_norm: bool = argument(
        default=False,
        help='apply layer normalization to the embedding outputs'
    )
    decoder_embed_norm: bool = argument(
        default=False,
        help='apply layer normalization to the embedding outputs'
    )
    rms_norm: bool = argument(
        default=False,
        help='use RMSNorm instead of LayerNorm'
    )
    norm_eps: float = argument(
        default=1e-5,
        help='epsilon parameter of normalization layers'
    )
    norm_bias: bool = argument(
        default=True,
        help='whether layer normalization should have a bias'
    )
    shared_norm: bool = argument(
        default=False,
        help="apply the same layer norm to the self attention and FFN blocks"
    )
    parallel_attention: bool = argument(
        default=False,
        help="compute self attention and FFN outputs in parallel"
    )
    encoder_positional_encoding: str = argument(
        default='sinusoidal',
        choices=['learned', 'sinusoidal', 'alibi', 'rotary', 't5'],
        help='type of positional encoding used in the encoder'
    )
    decoder_positional_encoding: str = argument(
        default='sinusoidal',
        choices=['learned', 'sinusoidal', 'alibi', 'rotary', 't5'],
        help='type of positional encoding used in the decoder'
    )
    alibi_max_bias: int = argument(
        default=8,
        help='maximum bias value in alibi positional embeddings'
    )
    rope_base: int = argument(
        default=10000,
        help='hyper-parameter of rotary positional embeddings'
    )
    max_qkv: Optional[float] = argument(
        help='clamp attention queries, keys and values to this maximum value'
    )
    positional_encoding_shift: int = argument(
        default=2,
        help='shift positional embeddings by this many positions'
    )
    shift_encoder_layers: Optional[int] = argument(
        help='when loading a checkpoint for training, shift the encoder layers by N from bottom to top. For example, '
             'if N=2, initialize the first two layers at random and initialize the 3rd layer with the parameters of '
             'the checkpoint\'s 1st layer. N can be negative'
    )
    shift_decoder_layers: Optional[int] = argument(
        help='same as --shift-encoder-layers but for the decoder'
    )
    checkpoint_activations: bool = argument(
        default=False,
        help='save GPU memory by not storing activations during the forward pass (this will slow down training as '
             'those will need to be recomputed during the backward pass)'
    )  # TODO: implement partial checkpointing (keep activations of "slow" layers like FFNs)
    model_type: Optional[str] = argument(
        choices=['encoder_decoder', 'decoder'],
        defaults={
            'language_modeling': 'decoder',
            'translation': 'encoder_decoder',
        },
        help='Type of model: encoder-decoder (e.g., T5), decoder-only (e.g., GPT), or encoder-only (e.g., BERT)'
    )
    prompt_loss: float = argument(
        default=1.0,
        help="multiplier for the prompt tokens in the training loss (default: 1.0), set to zero to disable prompt loss"
    )
    scale_embed: bool = argument(
        default=True,
        help='scale the embeddings by square root of their dimension'
    )
    embed_dropout: Optional[float] = argument(
        help='apply this amount of dropout to the embeddings (default: same as --dropout)'
    )
    encoder_max_len: int = argument(
        default=256,
        help='maximum number of positions in the encoder (can be used for positional embeddings), used to set the '
             'default maximum source length of the data'
    )
    decoder_max_len: Optional[int] = argument(
        defaults={
            'language_modeling': 1024,
            'translation': 256,
        },
        help='maximum number of positions in the decoder (can be used for positional embeddings), used to set the '
             'default maximum target length of the data'
    )
    disable_bos: bool = argument(
        default=False,
        help='decoder inputs start directly with the first target token instead of the beginning of sequence token'
    )
    lora_rank: int = argument(
        default=0,
        help='train LoRA adapters with this bottleneck dimension (the other parameters are frozen)'
    )
    lora_alpha: int = argument(
        default=8,
        help='scale the outputs of the low-rank adapters by this amount (hyper-parameter of LoRA)'
    )

    def setup_for_inference(self, cfg: DecodingAPIConfig):
        """
        Modify this model configuration according to given decoding configuration: while most model hyper-parameters
        come from the model checkpoint, some may be overriden at decoding
        """
        if cfg.model_args:
            for name, value in json.loads(cfg.model_args).items():
                setattr(self, name, value)

        self.shift_encoder_layers = None
        self.shift_decoder_layers = None
        self.lora_rank = 0  # lora weights are automatically added to the linear weights when loading the checkpoint
        self.set_defaults(cfg.task)
        assert self.decoder_max_len > cfg.max_output_len, (
            "--max-output-len cannot be higher than the model's max target length"
        )


@register_model_config('adapter_transformer')
class AdapterTransformerConfig(TransformerConfig):
    _arch: str = 'adapter_transformer'

    encoder_adapter_dim: int = argument(
        default=64,
        help='bottleneck dimension of the encoder adapters'
    )
    decoder_adapter_dim: int = argument(
        default=64,
        help='bottleneck dimension of the decoder adapters'
    )
    encoder_adapter_layer_ids: Optional[list[int]] = argument(
        help='zero-indexed indices of the encoder layers that should have adapters (default: all)'
    )
    decoder_adapter_layer_ids: Optional[list[int]] = argument(
        help='zero-indexed indices of the decoder layers that should have adapters (default: all)'
    )
    encoder_adapters: Optional[list[str]] = argument(
        help="manually set the names of the encoder adapters (default: 'default'). Multiple values will result in "
             "stacked adapters. An empty value will result in encoder adapters being disabled."
    )
    decoder_adapters: Optional[list[str]] = argument(
        help="manually set the names of the decoder adapters (default: 'default'). Multiple values will result in "
             "stacked adapters. An empty value will result in decoder adapters being disabled."
    )
    encoder_adapters_by: list[str] = argument(
        default=[],
        choices=['lang', 'source_lang', 'target_lang', 'domain'],
        help='train encoder adapters that are specific to the given metadata key. Different adapters will be activated '
             'depending on the languages or domain of the current batch. Note that this automatically adds this key to '
             '--batch-by, so that batches are homogeneous with respect to it'
    )
    decoder_adapters_by: list[str] = argument(
        default=[],
        choices=['lang', 'source_lang', 'target_lang', 'domain'],
        help='train decoder adapters that are specific to the given metadata key. Different adapters will be activated '
             'depending on the languages or domain of the current batch. Note that this automatically adds this key to '
             '--batch-by, so that batches are homogeneous with respect to it'
    )
    adapter_zero_init: bool = argument(
        default=False,
        help='initialize the adapter parameters to zero. This is useful at decoding when over-specifying the model '
             'with adapters that might not appear in the model checkpoint. By default, they are initialized to small '
             'values so that their output is close to identity'
    )
    train_all_params: bool = argument(
        default=False,
        help='do not freeze the other parameters (by default only the adapters are trained)'
    )

    def setup_for_inference(self, cfg: DecodingAPIConfig):
        super().setup_for_inference(cfg)
        # Manual adapter definition overriding the model's checkpoint configuration.
        # If several adapters are defined, they will be stacked in the same order.
        # You might want to combine that with --flexible.
        self.encoder_adapters = cfg.encoder_adapters or []
        self.decoder_adapters = cfg.decoder_adapters or []
        if cfg.encoder_adapters is not None or cfg.decoder_adapters is not None:
            self.encoder_adapters_by = []
            self.decoder_adapters_by = []
        
        self.adapter_zero_init = True
        # If some adapters are not in the checkpoint and --flexible is set, those will be just bypassed.
        # This enables the user to manipulate checkpoints to define custom adapter configurations that cannot be 
        # defined with options.
        self.encoder_adapter_layer_ids = None
        self.decoder_adapter_layer_ids = None


@register_model_config('hybrid_transformer')
class HybridTransformerConfig(TransformerConfig):
    _arch: str = 'hybrid_transformer'
    decoder_layers: int = 2

    decoder_hidden_size: int = argument(
        default=512,
        help='hidden size of the LSTMs'
    )
    decoder_embed_proj: bool = argument(
        default=False,
        help="apply a linear projection to the decoder's input embeddings before passing them to the first LSTM"
    )


@register_model_config('adapter_hybrid_transformer')
class AdapterHybridTransformerConfig(AdapterTransformerConfig):
    _arch: str = 'adapter_hybrid_transformer'
    decoder_layers: int = 2
    
    decoder_hidden_size: int = argument(
        default=512,
        help='hidden size of the LSTMs'
    )
    decoder_embed_proj: bool = argument(
        default=False,
        help="apply a linear projection to the decoder's input embeddings before passing them to the first LSTM"
    )


@register_model_config('moe_transformer')
class MOETransformerConfig(TransformerConfig):
    _arch: str = 'moe_transformer'

    encoder_expert_count: Union[int, dict] = argument(
        default=4,
        help='number of experts per encoder layer (can also be a dict specifying a different count per layer id)'
    )
    decoder_expert_count: Union[int, dict] = argument(
        default=4,
        help='number of experts per decoder layer (can also be a dict specifying a different count per layer id)'
    )
    encoder_expert_dim: Optional[int] = argument(
        default=None,
        help='bottleneck dimension of the encoder experts (default: same as --encoder-ffn-dim)'
    )
    decoder_expert_dim: Optional[int] = argument(
        default=None,
        help='bottleneck dimension of the decoder experts (default: same as --decoder-ffn-dim)'
    )
    encoder_expert_layer_ids: Optional[list[int]] = argument(
        default=None,
        help='use experts at these encoder layers (indices started at zero)'
    )
    decoder_expert_layer_ids: Optional[list[int]] = argument(
        default=None,
        help='use experts at these decoder layers (indices started at zero)'
    )
    encoder_expert_interval: int = argument(
        default=1,
        help='use experts at every Nth encoder layer (default: all layers)'
    )
    decoder_expert_interval: int = argument(
        default=1,
        help='use experts at every Nth decoder layer (default: all layers)'
    )
    moe_impl: str = argument(
        default='basic',
        choices=['basic', 'fused', 'tutel'],
        help='which Mixture-of-Experts implementation to use'
    )
    capacity_factor: float = argument(
        default=0.0,
        help="defines the maximum load of each expert (max_tokens = 2 * factor * batch_size / expert_count); the "
             "default value of 0 corresponds to an unlimited load"
    )
    load_balancing: float = argument(
        default=0.0,
        help="scale the load balancing loss term by this factor (default: no load balancing)"
    )


class TrainingConfig(DistributedConfig, TrackerConfig, EvalConfig, DecodingConfig):
    # Those won't be registered as arguments: they have to be manually set
    dataset_cfg: Optional[TrainingDatasetConfig] = None
    model_cfg: Optional[TransformerConfig] = None
    device: Optional[str] = None
    task_cfg: Optional[TaskConfig] = None

    lr: float = argument(
        default=0.0005,
        help='maximum learning rate. The learning rate starts at INIT_LR, then linearly increases for the first WARMUP '
             'steps until it reaches LR, then it decreases following the inverse square root of the update number'
    )
    adam_betas: list[float] = argument(
        default=(0.9, 0.999),
        nargs=2,
        help='hyper-parameters of the Adam optimizer'
    )
    warmup: int = argument(
        default=4000,
        help='number of training updates for the warmup phase, which linearly increases the learning rate up to its '
             'maximum value (specified by --lr). If warmup is 0, then linear decay is applied to MIN_LR in MAX_STEPS '
             'training steps'
    )
    init_lr: float = argument(
        default=0.0,
        help='initial learning rate in the warmup phase'
    )
    min_lr: float = argument(
        default=0,
        help='minimum value of the learning rate, after which is stops decaying and remains constant'
    )
    weight_decay: float = argument(
        default=0.0,
        help='weight decay value in Adam'
    )
    clip_norm: float = argument(
        default=1.0,
        help='clip the total gradient norm to this value. Normalize all gradients so that their norm does not exceed '
             'CLIP_NORM'
    )
    reset: bool = argument(
        default=False,
        help='ignore any existing model checkpoint and train from scratch. Also overwrites existing log files'
    )
    reset_optimizer: bool = argument(
        default=False,
        help="when loading a checkpoint, only load its models parameters, but reset training metrics and the states of "
             "the LR scheduler and of the optimizer (this takes precedence over --continue)"
    )
    flexible: bool = argument(
        default=False,
        help='load the checkpoint anyway if it has missing or unexpected parameters'
    )
    amp: bool = argument(
        default=False,
        help="use Pytorch's built-in automatic mixed precision (slower than the default fairseq-style mixed precision)"
    )
    virtual_dp_size: int = argument(
        default=1,
        help="accumulate gradients over this many batches. This is a way to 'artificially' increase the batch size "
             "without using more memory. There are two main advantages: the learning rate can be increased and there "
             "is less GPU synchronization. This value is normalized by the number of GPUs so that "
             "'--virtual-dp-size 4' on 1, 2 or 4 GPUs always gives the same batch size"
    )
    find_unused_parameters: bool = argument(
        default=False,
        help='whether GPUs may use different model parameters in a given update'
    )
    flat_fp16: bool = argument(
        default=False,
        help='speed up training at the cost of higher memory usage by having Adam work on flattened gradients'
    )
    memory_efficient_fp16: bool = argument(
        default=True,
        help='save GPU memory at the cost of slightly slower training by converting the float16 gradients to float32 '
             'only when needed (note that this implies --no-flat-fp16)'
    )
    fsdp: bool = argument(
        default=False,
        help='apply the FSDP algorithm (Fully Sharded Data Parallel), i.e., shard the full model & optimizer states '
             'across all GPUs, which effectively divides their memory usage by the number of GPUs, at the cost of '
             'higher communication time'
    )
    reset_params_regex: Optional[str] = argument(
        help='ignore parameters from checkpoint whose name matches this regex (note that this may require --flexible)'
    )
    config: Optional[str] = argument(
        aliases=['-c'],
        help='path to a YAML configuration file. Parameters defined there have higher precedence than Pasero defaults '
             'and lower precedence than command-line parameters. Some parameters cannot be specified as precisely by '
             'the command line (e.g., multilingual or multi-domain training and validation corpora)'
    )
    data_dir: Optional[str] = argument(
        help='path to the directory containing the training and validation data, dictionaries and tokenizers '
             '(required)'
    )
    model_dir: Optional[str] = argument(
        aliases=['-o'],
        help='directory where the model checkpoints and training logs will be saved (required) It is automatically '
             'created if necessary. Some files will also be automatically copied from DATA_DIR to MODEL_DIR '
             '(e.g., dictionaries and tokenizers)'
    )
    train_corpora: list[Union[str, dict]] = argument(
        default=['train'],
        nargs='+',
        help="list of training corpus prefixes (e.g., 'train', 'train.de-en', 'train.{pair}' or 'train.{src}-{tgt}'), "
             "relative to DATA_DIR or absolute. YAML config can define corpora as dicts to specify more attributes "
             "(e.g., 'source_langs', 'target_langs', 'domain', etc.)"
    )
    valid_corpora: list[Union[str, dict]] = argument(
        default=['valid'],
        nargs='+',
        help="list of validation corpus prefixes (e.g., 'valid', 'valid.de-en', 'valid.{pair}' or "
             "'valid.{src}-{tgt}'), relative to DATA_DIR or absolute. YAML config can define corpora as dicts to "
             "specify more attributes (e.g., 'source_langs', 'target_langs', 'domain', etc.)"
    )
    ckpt: Optional[str] = argument(
        help='checkpoint to restore (if it exists). If it is just a name, it is assumed to be a file by that name in '
             'MODEL_DIR. A path is interpreted as relative to the working directory. Not that unless --reset is set, '
             "if MODEL_DIR already contains 'model_last.bin', it will be loaded and --ckpt will have no effect"
    )
    continue_: bool = argument(   # parses --continue
        default=False,
        help="continue training from CKPT without resetting the optimizer and step number. This is activated by "
             "default if MODEL_DIR already contains a checkpoint"
    )
    arch: str = argument(
        default='transformer',
        help='model architecture (e.g., transformer, transformer_big or adapter_transformer)'
    )
    max_steps: Optional[int] = argument(
        help='maximum number of training updates (required). Note that an update corresponds to approximately '
             'max(WORLD_SIZE, VIRTUAL_DP_SIZE) * MAX_TOKENS source or target tokens. Actual numbers of tokens per batch '
             '(wpb) or line pairs per batch (bsz) are logged during training')
    valid_interval: Optional[int] = argument(
        help='number of training updates before validation (required), must be a multiple of --log-interval and '
             '--save-interval'
    )
    log_interval: int = argument(
        default=100,
        help='number of training updates before logging metrics. The same interval will be used for sending values to '
             'the experiment tracker (if applicable). Note that this value can influence the value of the training '
             'metrics, which are running averages over a window of this size (e.g., ppl, loss)'
    )
    log_file: Optional[str] = argument(
        help="path for the logging output relative to MODEL_DIR (default: 'train.log')"
    )
    save_interval: Optional[int] = argument(
        help='number of training updates before saving a model checkpoint, must be a multiple of --log-interval '
             '(default: same value as --valid-interval)'
    )
    save_initial_checkpoint: bool = argument(
        default=False,
        help="save the initial weights (after random init and checkpoint loading and before starting to train) as "
             "'model_init.bin'"
    )
    save_trainable_only: bool = argument(
        default=False,
        help='only save the trainable parameters (e.g., adapters) of the model, assuming the frozen parameters are '
             'available elsewhere'
    )
    keep_interval: Optional[int] = argument(
        help='keep all checkpoints whose update count is a multiple of this value, must be a multiple of '
             '--save-interval. By default only the last and best checkpoints are kept'
    )
    keep_last: int = argument(
        default=1,
        help='how many last checkpoints to keep'
    )
    average_checkpoints: bool = argument(
        default=False,
        help='average the last checkpoints before validation'
    )
    validate_at_start: bool = argument(
        default=False,
        help='run validation once before starting training'
    )
    only_validate: bool = argument(
        default=False,
        help='only run validation and disable training'
    )
    benchmark: bool = argument(
        default=False,
        help='compute the time spent in each Transformer component (this will slow down training)'
    )
    verbose: bool = argument(
        aliases=['-v'],
        default=False,
        help='log training and validation data examples'
    )
    freeze_params_regex: Optional[str] = argument(
        help="freeze any parameter that matches this regex and unfreeze any that does not match it. This overrides the "
             "default behavior of model architectures (e.g., adapter_transformer)"
    )
    train_params_regex: Optional[str] = argument(
        help="freeze any parameter that doesn't match this regex and unfreeze any that matches it. This overrides the "
             "default behavior of model architectures (e.g., adapter_transformer)"
    )
    task: str = argument(
        choices=[
            'translation',
            'speech_translation',
            'language_modeling',
            'doc_level_translation',
            'dialogue',
        ],
        default='translation',
    )
    dataset_type: str = argument(
        choices=['dynamic', 'simple', 'debug'],
        default='dynamic',
        help="type of dataset: 'dynamic' (default) for on-the-fly data loading and preprocessing in a single pipeline; "
             "or 'simple' to avoid using queues and shard the data across N workers per GPU, each doing their own "
             "batching"
    )
    debug: bool = argument(
        default=False,
        help="activate debug mode: will allow breakpoints in the preprocessing code and in multi-GPU training runs. "
             "This also activates --verbose. This may slow down training and should be used for debugging only."
    )
    early_stopping_metric: Optional[str] = argument(
        defaults={
            'language_modeling': 'nll_loss',
            'translation': 'chrf',
        },
        choices=evaluation.METRICS + ['nll_loss'],
        help='which metric to use to select the best checkpoint. Note that by default, the scores all validation sets '
             "are averaged. A validation corpus can be excluded from this average by setting its 'early_stopping' "
             "property to False"
    )
    patience: Optional[int] = argument(
        help='stop training when the validation score has not improved in the last N evaluations'
    )
    patience_min_steps: int = argument(
        default=0,
        help='start losing patience after this many steps'
    )
    expected_scores: list[dict] = argument(
        default=[],
        help='expected scores for this training run (this should be specified in the YAML config and not as a command '
             'line parameter)'
    )

    def __init__(self, *opts, strict: bool = True, **kwargs):
        """
        Training options can be specified in 3 different ways (from lowest to highest precedence):

        - YAML config file (given as 'config' argument via `opts` or `kwargs`)
        - keyword arguments (given via `kwargs`)
        - command line arguments (given via `opts` or obtained automatically from `sys.argv`)
        """
        opts = opts or sys.argv[1:]
        super().__init__(*opts, strict=False, **kwargs)  # to get '--config' and parse the YAML config:
        # we need to read the config file to pick up the values of the '--arch' and '--dataset-type' options,
        # because they are required by `get_model_config` and `get_dataset_config`

        self.parse_args(strict=False)  
        yaml_opts = yaml.safe_load(open(self.config)) if self.config else {}
        yaml_opts = self.parse_dict(yaml_opts, strict=False)
        kwargs = self.parse_dict(kwargs, strict=False)
        cli_opts = self.parse_args(opts, strict=False)  # has precedence over YAML config

        if self.debug:  # debug mode that lets us put breakpoints anywhere
            self.dataset_type = 'debug'  # other datasets use multiprocessing, which is incompatible with breakpoints
            self.verbose = True

        dataset_cfg = get_dataset_config(self.dataset_type)
        model_cfg = get_model_config(self.arch)
        task_cfg = get_task_config(self.task)

        parse_help(self, task_cfg, dataset_cfg, model_cfg)

        yaml_opts = task_cfg.parse_dict(yaml_opts, strict=False)
        kwargs = task_cfg.parse_dict(kwargs, strict=False)
        cli_opts = task_cfg.parse_args(cli_opts, strict=False)

        yaml_opts = dataset_cfg.parse_dict(yaml_opts, strict=False)
        kwargs = dataset_cfg.parse_dict(kwargs, strict=False)
        cli_opts = dataset_cfg.parse_args(cli_opts, strict=False)
        
        yaml_opts = model_cfg.parse_dict(yaml_opts, strict=strict)  # all YAML config options should be accounted for
        kwargs = model_cfg.parse_dict(kwargs, strict=strict)
        cli_opts = model_cfg.parse_args(cli_opts, strict=strict)  # all command-line options should be accounted for

        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.task_cfg = task_cfg

        # set default options that depend on the task
        for cfg in self, self.dataset_cfg, self.model_cfg:
            cfg.set_defaults(self.task)

        self.finalize()

    def as_dict(self) -> dict:
        dict_ = super().as_dict()
        if self.dataset_cfg is not None:
            dict_.update(self.dataset_cfg.as_dict())
        if self.model_cfg is not None:
            dict_.update(self.model_cfg.as_dict())
        if self.task_cfg is not None:
            dict_.update(self.task_cfg.as_dict())
        return dict(sorted(dict_.items()))

    def finalize(self):
        assert self.model_cfg is not None
        assert self.dataset_cfg is not None

        # Not using the "required" argument of argparse because these options can also be defined in the YAML config
        assert self.data_dir, '--data-dir is required'
        assert self.model_dir, '-o/--model-dir is required'

        # --data-dir and --model-dir values may contain placeholders (e.g., {src}, {tgt}, {pair} or {lang}),
        # replace those by the task's languages
        for opt in 'data_dir', 'model_dir', 'tracker_project_name', 'tracker_run_name', 'ckpt':
            value = getattr(self, opt, None)
            if value:
                value = self.task_cfg.replace_placeholders(value)
                setattr(self, opt, value)

        max_len = self.task_cfg.set_max_length(self.model_cfg)
        self.task_cfg.finalize()

        assert self.max_steps is not None, '--max-steps is required'
        assert self.valid_interval, '--valid-interval is required'
        if not self.save_interval:
            self.save_interval = self.valid_interval
        assert os.path.isdir(self.data_dir), 'data directory does not exist'
        assert self.max_steps == 0 or self.only_validate or self.early_stopping_metric in self.metrics + ['nll_loss']
        assert self.valid_interval % self.log_interval == 0, 'valid interval must be a multiple of logging interval'
        assert self.save_interval % self.log_interval == 0, 'save interval must be a multiple of logging interval'
        assert self.valid_interval % self.save_interval == 0, 'valid interval must be a multiple of save interval'
        assert not self.keep_interval or self.keep_interval % self.save_interval == 0, \
            'keep interval must be a multiple of save interval'
        assert not (self.fsdp and self.average_checkpoints), '--average-checkpoints is not compatible with --fsdp'

        if self.sequence_parallel and self.tp_size > 1:  # with sequence parallelism, only the master rank reads 
            # batches, then it shards them and scatters them to the other ranks. But for the sharding to work, the
            # batch size (in terms of sequences per batch) should be a multiple of the number of GPUs: which requires
            # --batch-size-multiple to be correctly set, and --batch-size (the maximum number of tokens in a batch) to
            # be large enough
            if self.task_cfg.batch_size_multiple is None:
                self.task_cfg.batch_size_multiple = self.tp_size
            else:
                assert self.task_cfg.batch_size_multiple % self.tp_size == 0, 'tensor parallelism with ' \
                    '--sequence-parallel requires --batch-size-multiple to be a multiple of --tp-size'
            assert self.task_cfg.batch_size >= (max_len * self.task_cfg.batch_size_multiple), 'batch size is too ' \
                'small for --sequence-parallel (it should be a least max sequence length * batch size multiple)'

        if self.only_validate:
            self.max_steps = 0
            self.validate_at_start = True
            self.reset_optimizer = True
            self.log_file = self.log_file or 'valid.log'
        else:
            self.log_file = self.log_file or 'train.log'

        if self.seed == -1 or self.seed is None:
            # pick the seed at random (default): more reproducible than not setting any seed at all
            # ("cfg" is logged later)
            self.seed = random.randrange(10**9)

        for name in 'train_corpora', 'valid_corpora':
            # Corpora can be defined as lists of corpus prefixes (via the command line) or as list of corpus 
            # dictionaries defining all their properties: corpus prefix, languages, domain, etc. (via the YAML config).
            # Convert these two in the same dictionary format:
            corpora = getattr(self, name)
            if corpora is None:
                continue
            for i, corpus in enumerate(corpora):
                corpus = corpora[i]
                if isinstance(corpus, str):
                    corpora[i] = {'paths': [corpus]}
                else:
                    assert isinstance(corpus, dict)

        if self.dataset_cfg.cache_line_index and self.dataset_cfg.line_index_path is None:
            # Automatically set the location of the line position cache, based on the value of 'data_dir', e.g.:
            # '/some_disk/data/ParaCrawl' -> 'tmp/some_disk_data_ParaCrawl_index.bin'
            # This assumes that the local 'tmp' directory is a symlink to a directory on a shared filesystem, as 
            # suggested in README.md
            data_dir = os.path.realpath(self.data_dir)
            index_name = data_dir.replace('/', '_')
            index_name = f'{index_name}_index.bin'.strip('_')
            tmp_dir = os.environ.get('PASERO_TMP') or 'tmp'
            self.dataset_cfg.line_index_path = os.path.join(tmp_dir, index_name)

    @property
    def inference_options(self):
        """
        Convert this configuration to an inference configuration, which will be saved as 'inference.yaml' in the model
        directory and used as default by `pasero-decode` or `TextGenerator`.
        Contrary to `training.yaml`, this contains only the non-default values.
        Note that `task.inference_options` will need to be called as well to include the task-specific and 
        preprocessing options.
        """
        options = {}
        if self.save_trainable_only and self.ckpt:
            options['other_ckpt'] = [self.ckpt]
        options['dtype'] = self.dtype
        decoding_cfg = DecodingConfig(self)
        default_decoding_cfg = DecodingConfig()
        for name, value in decoding_cfg.as_dict().items():
            default_value = getattr(default_decoding_cfg, name)
            if value != default_value:
                options[name] = value
        return options


class TranslationTaskConfig(TaskConfig):
    source_lang: Optional[str] = argument(
        aliases=['-s'],
        help='source language'
    )
    target_lang: Optional[str] = argument(
        aliases=['-t'],
        help='target language'
    )
    source_langs: Optional[list[str]] = argument(
        nargs='+',
        help='source languages covered by the model. At training, the language pairs are the cartesian product of '
             'SOURCE_LANGS and TARGET_LANGS (-s de fr -t en es -> de-en de-es fr-en fr-es). In the YAML configuration, '
             'each corpus can specify a different list of source languages'
    )
    target_langs: Optional[list[str]] = argument(
        nargs='+',
        help='target languages covered by the model. At training, the language pairs are the cartesian product of '
             'SOURCE_LANGS and TARGET_LANGS (-s de fr -t en es -> de-en de-es fr-en fr-es). In the YAML configuration, '
             'each corpus can specify a different list of target languages'
    )
    lang_pairs: Optional[list[str]] = argument(
        aliases=['-l'],
        nargs='+',
        help='language pairs for the training, validation or inference corpora. It is a more fine-grained way to '
             'specify language pairs than with SOURCE_LANGS and TARGET_LANGS and it has priority over the latter. In '
             'the YAML configuration, each corpus can specify a different list language pairs.'
    )
    allow_monolingual: bool = argument(
        default=False,
        help='allow monolingual language pairs (e.g., fr-fr), which are skipped by default'
    )
    valid_source_langs: Optional[list[str]] = argument(
        nargs='+',
        help='source languages for the validation corpora. This option is only allowed at training'
    )
    valid_target_langs: Optional[list[str]] = argument(
        nargs='+',
        help='target languages for the validation corpora. This option is only allowed at training'
    )
    valid_lang_pairs: Optional[list[str]] = argument(
        nargs='+',
        help='language pairs for the validation corpora. This option is only allowed at training'
    )
    max_source_len: int = argument(
        help='maximum tokens per source line. Longer lines will be truncated (validation data) or skipped '
            '(training data). Default: same as --encoder-max-len'
    )
    max_target_len: int = argument(
        help='maximum tokens per target line. Longer lines will be truncated (validation data) or skipped '
             '(training data). Default: same as --decoder-max-len'
    )
    min_len_ratio: Optional[float] = argument(
        help='skip line pairs whose source/target length ratio is lower than this value (incompatible with --truncate)'
    )
    max_len_ratio: Optional[float] = argument(
        help='skip line pairs whose source/target length ratio is higher than this value (incompatible with --truncate)'
    )
    escape_emojis: bool = argument(
        default=False,
        help='at inference, replace source-side emojis with placeholders and then output placeholders with those emojis'
    )
    copy_placeholder: bool = argument(
        default=True,
        help='replace OOV symbols that appear on both the source and target side (with the same count) by a special '
             'copy token, instead of the usual <unk>'
    )

    # Tagging
    source_tags: Optional[list[str]] = argument(
        help='prefix every source sentence with these special tokens (note that different tags can be specified per '
             'corpus in the YAML configuration)'
    )
    target_tags: Optional[list[str]] = argument(
        help='prefix every target sentence with these special tokens (note that different tags can be specified per '
             'corpus in the YAML configuration)'
    )
    # TODO: Allow placeholders for maximum flexibility: {domain} {lang} {target_lang} {source_lang}
    source_lang_code: bool = argument(
        default=False,
        help="prefix source lines with the source language code (e.g., '<lang:de>')"
    )
    target_lang_code: bool = argument(
        default=False,
        help='prepend target lines with the target language code. When decoding, the model will be forced to generate '
             'this token at the first time step'
    )
    lang_code: bool = argument(
        default=False,
        help="prefix source lines with the target language code (e.g., '<lang:en>')"
    )
    domain_tag: bool = argument(
        default=False,
        help="prefix source lines with a domain tag (e.g., '<domain:medical>'). At training time, the tag is defined "
             "thanks to the 'domain' property of each corpus. At test time, it is defined with --domain"
    )

    # Target-side tokenization
    target_dict: Optional[str] = argument(
        help="path to the target dictionary, absolute or relative to DATA_DIR (at training) or MODEL_DIR "
             "(at inference). Default value: same as --dict. Leave empty if --shared-embeddings is set."
    )
    target_tokenizer: Optional[str] = argument(
        help='BPE implementation to use for the target lines (default: same as --tokenizer)'
    )
    target_tokenizer_path: Optional[str] = argument(
        help='path to the BPE model used to tokenize target lines (default: same as --tokenizer-path)'
    )
    target_tokenizer_vocab: Optional[str] = argument(
        help='path to the target BPE vocabulary used for frequency-based filtering (default: same as --tokenizer-vocab)'
    )
    target_vocabulary_threshold: Optional[int] = argument(
        help='minimum frequency of generated subwords on the target side (default: same as --vocabulary-threshold)'
    )
    target_spell_out: float = argument(
        default=0.0,
        help='apply BPE dropout to the target training data with this rate (default: disabled)'
    )
    target_bpe_dropout: float = argument(
        default=0.0,
        help='spell out training target words with this probability (default: disabled)'
    )
    old_source_dict: Optional[str] = argument(
        help='relative path to a dictionary that should be used to re-map the source embeddings (can be used for '
             'test-time vocabulary filtering)'
    )
    old_target_dict: Optional[str] = argument(
        help='relative path to a dictionary that should be used to re-map the target embeddings (can be used for '
             'test-time vocabulary filtering)'
    )
    default_embed: Optional[str] = argument(
        default='<unk>',
        help='use the embedding of this symbol to initialize the embeddings of unknown words when re-mapping '
             'vocabularies'
    )
    freeze_source_embed_regex: Optional[str] = argument(
        help='freeze all source embeddings whose token matches this regex'
    )

    @classmethod
    def format_path(cls, path: str, source_lang: str, target_lang: str) -> str:
        return (
            path
            .replace('{src}', source_lang)
            .replace('{tgt}', target_lang)
            .replace('{pair}', f'{source_lang}-{target_lang}')
        )

    def replace_placeholders(self, path: str) -> str:
        """
        Fill in the {src}, {tgt} and {pair} placeholders in the config paths (e.g., data dir, model dir, bpe codes,
        etc.) with the first lang pair (from --lang-pairs) or source lang and target lang (from -s/--source-lang and 
        -t/--target-langs).
        """
        if self.lang_pairs:
            source_lang, target_lang = self.lang_pairs[0].split('-')
        else:
            source_lang = self.source_lang or 'src'
            target_lang = self.target_lang or 'tgt'
        return self.format_path(path, source_lang, target_lang)

    def finalize(self):
        super().finalize()
        # There are several ways to define source languages:
        # - a single source language with -s/--source-lang
        # - a list of source languages with --source-langs
        # - a list of language pairs with -l/--lang-pairs
        # At inference, `source_lang` can be used to define the default source language (in 'inference.yaml') and 
        # `source_langs` to define the full list of covered languages.
        # At training, `source_langs` and `target_langs` are transformed into language pairs by doing their cartesian 
        # product.
        if self.source_lang:
            # infer --source-langs from --source-lang
            if not self.source_langs:
                self.source_langs = [self.source_lang]
            elif self.source_lang not in self.source_langs:
                # the language is theoretically not covered, but we are explicitely passing it
                self.source_langs.append(self.source_lang)
        elif self.source_langs and len(self.source_langs) == 1:  # if -s/--source-lang is not given but the list of 
            # covered source languages (--source-lang) has only one element, there is not ambiguity and --source-lang 
            # (i.e., the default source language) can be inferred
            self.source_lang = self.source_langs[0]
        # Same for target languages
        if self.target_lang:
            if not self.target_langs:
                self.target_langs = [self.target_lang]
            elif self.source_lang not in self.target_langs:
                self.target_langs.append(self.target_lang)
        elif self.target_langs and len(self.target_langs) == 1:
            self.target_lang = self.target_langs[0]
        
        for opt in 'target_tokenizer_path', 'target_tokenizer_vocab', 'target_dict':
            value = getattr(self, opt, None)
            if value:
                setattr(self, opt, self.replace_placeholders(value))

    def set_max_length(self, model_cfg: 'TransformerConfig') -> int:
        """
        Automatically set the maximum source and target length for the data (if they are not set) according to the
        encoder and decoder's maximum positions.

        Returns the overall max length: `max(max_source_len, max_target_len)`
        """

        if self.max_target_len:
            assert self.max_target_len <= model_cfg.decoder_max_len
        else:
            self.max_target_len = model_cfg.decoder_max_len

        if model_cfg.model_type == 'decoder':
            if self.max_source_len:
                assert self.max_source_len < self.max_target_len
            else:
                self.max_source_len = self.max_target_len // 2  # there is no encoder, but we still
                # have source sentences that will be used as prompt to the decoder: allocate half the decoder's max
                # length to the source and half to the target/output
            max_len = self.max_target_len
            
        else:
            if self.max_source_len:
                assert self.max_source_len <= model_cfg.encoder_max_len
            else:
                self.max_source_len = model_cfg.encoder_max_len
            max_len = max(self.max_source_len, self.max_target_len)
        
        assert self.batch_size >= max_len, 'batch size should be at least as high as the maximum sequence length'
        return max_len


class DocumentLevelTranslationTaskConfig(TranslationTaskConfig):
    # document-level training
    max_doc_size: int = argument(
        default=1,
        help="consecutive sentences in ordered corpora (i.e., with the 'ordered' attribute set to True) will be "
             "concatenated into documents of up to this size (size is uniformly sampled between 1 and this max size)"
    )
    sent_merge_prob: float = argument(
        default=0.0,
        help="each sentence pair in a document will be concatenated to the previous sentence pair with this probability"
    )
    sent_sep: Optional[str] = argument(
        default='<sep>',
        help="use this special token as a separator between sentences in a document (it has to be in the dictionary)"
    )
    trailing_sent_sep: bool = argument(
        default=False,
        help='also add a sentence separator after the last sentence of a document'
    )


class LanguageModelingTaskConfig(TaskConfig):
    langs: Optional[list[str]] = argument(
        aliases=['-l', '-t', '--target-langs'],
        nargs='+',
        help='languages for the training and validation corpora'
    )
    valid_langs: Optional[list[str]] = argument(
        aliases=['--valid-target-langs'],
        nargs='+',
        help='languages for the validation corpora'
    )
    max_len: int = argument(
        help='maximum tokens per line. Longer lines will be truncated (validation data) or skipped (training data). '
             'Default: same as --decoder-max-len'
    )

    # Tagging
    tags: Optional[list[str]] = argument(
        help='prefix every sentence with these tokens (note that different tags can be specified per corpus in the '
             'YAML configuration)'
    )
    lang_code: bool = argument(
        default=False,
        help="prefix every sentence with a language code (e.g., '<lang:en>')"
    )
    domain_tag: bool = argument(
        default=False,
        help="prefix every sentence with a domain tag (e.g., '<domain:medical>'). At training time, the tag is defined "
             "thanks to the 'domain' property of each corpus. At test time, it is defined with --domain"
    )

    @classmethod
    def format_path(cls, path: str, lang: str) -> str:
        return path.replace('{lang}', lang)

    def replace_placeholders(self, path: str) -> str:
        lang = self.langs[0] if self.langs else 'tgt'
        return self.format_path(path, lang)

    def set_max_length(self, model_cfg: 'TransformerConfig') -> int:
        """
        Automatically set the maximum data length (if it is not set) according to the decoder's maximum positions
        """
        if self.max_len:
            assert self.max_len <= model_cfg.decoder_max_len
        else:
            self.max_len = model_cfg.decoder_max_len
        
        assert self.batch_size >= self.max_len, 'batch size should be at least as high as the maximum sequence length'
        return self.max_len


class DialogueTaskConfig(LanguageModelingTaskConfig):
    custom_chat_template: Optional[str] = argument(help="custom chat template specified in the HuggingFace format")
    chat_template: Optional[str] = argument(help="chat template to use (e.g., 'chatml')")
    system_prompt: Optional[str] = argument(help='message to use as system prompt, if the chat template supports it')


class NLLBTranslationTaskConfig(TranslationTaskConfig):
    expert_ckpt: Optional[list[str]] = argument(
        help='path to expert checkpoints'
    )
    expert_json: Optional[str] = argument(
        help='JSON file mapping language pairs to lists of expert checkpoints paths to use for those language pairs'
    )
    expert_dir: Optional[str] = argument(
        help='directory containing the expert checkpoints (whose names are specified with --expert-ckpt or '
             '--expert-json)'
    )


def get_dataset_config(name: str = 'dynamic', strict: bool = True) -> TrainingDatasetConfig:
    # maps dataset type names (as specified by --dataset-type) to dataset configs 
    configs = {
        'dynamic': DynamicTrainingDatasetConfig,
        'simple': SimpleDynamicTrainingDatasetConfig,
        'debug': DebugTrainingDatasetConfig,
    }
    if name not in configs:
        if strict:
            raise ValueError(f'unknown dataset type: {name}')
    return configs.get(name, DynamicTrainingDatasetConfig)()


def get_task_config_cls(name: str = 'translation', strict: bool = True) -> type[TaskConfig]:
    # maps task names (as specified by --task) to task config classes
    configs = {
        'translation': TranslationTaskConfig,
        'speech_translation': TranslationTaskConfig,
        'language_modeling': LanguageModelingTaskConfig,
        'doc_level_translation': DocumentLevelTranslationTaskConfig,
        'nllb_translation': NLLBTranslationTaskConfig,
        'dialogue': DialogueTaskConfig,
    }
    if name not in configs:
        if strict:
            raise ValueError(f'unknown task: {name}')
    return configs.get(name, TranslationTaskConfig)


def get_task_config(name: str = 'translation', strict: bool = True) -> TaskConfig:
    cls = get_task_config_cls(name, strict=strict)
    return cls()


def get_model_config(arch: str = 'transformer', strict: bool = True) -> TransformerConfig:
    if arch not in MODEL_CONFIGS:
        if strict:
            raise ValueError(f'unknown architecture: {arch}')
    config_cls = MODEL_CONFIGS[arch]
    return config_cls()


@register_model_config('transformer_big', 'transformer_wmt_en_de_big', 'transformer_vaswani_wmt_en_de_big')
class TransformerBigConfig(TransformerConfig):
    embed_dim: int = 1024
    encoder_ffn_dim: int = 4096
    decoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_attention_heads: int = 16

@register_model_config('transformer_wide')
class TransformerWideConfig(TransformerBigConfig):
    encoder_ffn_dim: int = 8192
    decoder_ffn_dim: int = 8192

@register_model_config('transformer_small', 'transformer_iwslt_de_en')
class TransformerSmallConfig(TransformerConfig):   # 'transformer_iwslt_de_en' in fairseq
    embed_dim: int = 512
    encoder_ffn_dim: int = 1024
    decoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 4
    decoder_attention_heads: int = 4

@register_model_config('adapter_hybrid_transformer_big')
class AdapterHybridTransformerBigConfig(AdapterHybridTransformerConfig):
    embed_dim: int = 1024
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_hidden_size: int = 1024

@register_model_config('adapter_hybrid_transformer_wide')
class AdapterHybridTransformerWideConfig(AdapterHybridTransformerBigConfig):
    encoder_ffn_dim: int = 8192
    decoder_hidden_size: int = 2048

@register_model_config('mbart_large')
class MBARTConfig(TransformerBigConfig):
    encoder_layers: int = 12
    decoder_layers: int = 12
    encoder_embed_norm: bool = True
    decoder_embed_norm: bool = True
    encoder_positional_encoding: str = 'learned'
    decoder_positional_encoding: str = 'learned'
    encoder_prenorm: bool = True
    decoder_prenorm: bool = True
    encoder_max_len: int = 1024
    decoder_max_len: int = 1024

@register_model_config('nllb_600m')
class NLLB600MConfig(TransformerBigConfig):
    encoder_layers: int = 12
    decoder_layers: int = 12
    encoder_prenorm: bool = True
    decoder_prenorm: bool = True

@register_model_config('nllb_1b3')
class NLLB1B3Config(NLLB600MConfig):
    encoder_layers: int = 24
    decoder_layers: int = 24
    encoder_ffn_dim: int = 8192
    decoder_ffn_dim: int = 8192

@register_model_config('nllb_3b3')
class NLLB3B3Config(NLLB1B3Config):
    embed_dim: int = 2048

@register_model_config('bloom_560m')
class Bloom560MConfig(TransformerConfig):
    decoder_layers: int = 24
    decoder_max_len: int = 2048
    model_type: str = 'decoder'
    decoder_positional_encoding: str = 'alibi'
    decoder_prenorm: bool = True
    embed_dim: int = 1024
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    scale_embed: bool = False
    decoder_embed_norm: bool = True
    activation_fn: str = 'gelu_tanh'
    disable_bos: bool = True

@register_model_config('bloom_1b1')
class Bloom1B1Config(Bloom560MConfig):
    embed_dim: int = 1536
    decoder_ffn_dim: int = 6144

@register_model_config('bloom_1b7')
class Bloom1B7Config(Bloom560MConfig):
    embed_dim: int = 2048
    decoder_ffn_dim: int = 8192

@register_model_config('bloom_3b')
class Bloom3BConfig(Bloom560MConfig):
    decoder_layers: int = 30
    embed_dim: int = 2560
    decoder_ffn_dim: int = 10240
    decoder_attention_heads: int = 32

@register_model_config('bloom_7b')
class Bloom7BConfig(Bloom560MConfig):
    decoder_layers: int = 30
    embed_dim: int = 4096
    decoder_ffn_dim: int = 16384
    decoder_attention_heads: int = 32

@register_model_config('llama_7b')  # Llama v1 or v2
class Llama7BConfig(TransformerConfig):
    decoder_layers: int = 32
    decoder_max_len: int = 4096
    model_type: str = 'decoder'
    decoder_positional_encoding: str = 'rotary'
    decoder_prenorm: bool = True
    tied_output_projection: bool = False
    embed_dim: int = 4096
    decoder_ffn_dim: int = 11008
    decoder_attention_heads: int = 32
    scale_embed: bool = False
    activation_fn: str = 'swiglu'
    rms_norm: bool = True
    has_bias: bool = False

@register_model_config('llama_13b')  # Llama v1 or v2
class Llama13BConfig(Llama7BConfig):
    decoder_layers: int = 40
    embed_dim: int = 5120
    decoder_ffn_dim: int = 13824
    decoder_attention_heads: int = 40

@register_model_config('llama_34b')  # Llama v2
class Llama34BConfig(Llama7BConfig):
    decoder_layers: int = 48
    embed_dim: int = 8192
    decoder_ffn_dim: int = 22016
    decoder_attention_heads: int = 64
    attention_heads_kv: int = 8

@register_model_config('llama_70b')  # Llama v2
class Llama70BConfig(Llama7BConfig):
    decoder_layers: int = 80
    embed_dim: int = 8192
    decoder_ffn_dim: int = 28672
    decoder_attention_heads: int = 64
    attention_heads_kv: int = 8

@register_model_config('llama_30b')  # Llama v1
class Llama30BConfig(Llama7BConfig):
    decoder_layers: int = 60
    embed_dim: int = 6656
    decoder_ffn_dim: int = 17920
    decoder_attention_heads: int = 52
    norm_eps: float = 1e-06
    decoder_max_len: int = 2048

@register_model_config('llama_65b')  # Llama v1
class Llama65BConfig(Llama7BConfig):
    decoder_layers: int = 80
    embed_dim: int = 8192
    decoder_ffn_dim: int = 22016
    decoder_attention_heads: int = 64
    decoder_max_len: int = 2048

@register_model_config('llama_3b')  # OpenLLaMA-3b-v2
class Llama3BConfig(Llama7BConfig):
    decoder_layers: int = 26
    embed_dim: int = 3200
    decoder_ffn_dim: int = 8640
    decoder_attention_heads: int = 32
    decoder_max_len: int = 2048
    norm_eps: float = 1e-06


@register_model_config('mistral_7b')
class Mistral7BConfig(Llama7BConfig):
    attention_heads_kv: int = 8
    decoder_ffn_dim: int = 14336
    sliding_window: int = 4096
    decoder_max_len: int = 32768


@register_model_config('mpt_7b')
class MPT7BConfig(TransformerConfig):
    decoder_layers: int = 32
    decoder_max_len: int = 2048
    model_type: str = 'decoder'
    decoder_positional_encoding: str = 'alibi'
    decoder_prenorm: bool = True
    embed_dim: int = 4096
    decoder_ffn_dim: int = 16384
    decoder_attention_heads: int = 32
    scale_embed: bool = False
    activation_fn: str = 'gelu'
    has_bias: bool = False
    norm_bias: bool = False
    disable_bos: bool = True

@register_model_config('mpt_7b_65k')
class MPT7B65kConfig(MPT7BConfig):
    alibi_max_bias: int = 16
    max_qkv: float = 6.0
    decoder_max_len: int = 65536

@register_model_config('mpt_30b')
class MPT30BConfig(MPT7BConfig):
    decoder_layers: int = 48
    decoder_max_len: int = 8192
    embed_dim: int = 7168
    decoder_ffn_dim: int = 28672
    decoder_attention_heads: int = 64

@register_model_config('falcon_7b')
class Falcon7BConfig(TransformerConfig):
    decoder_layers: int = 32
    decoder_max_len: int = 2048
    model_type: str = 'decoder'
    decoder_positional_encoding: str = 'rotary'
    decoder_prenorm: bool = True
    embed_dim: int = 4544
    decoder_ffn_dim: int = 18176
    decoder_attention_heads: int = 71
    attention_heads_kv: int = 1
    scale_embed: bool = False
    activation_fn: str = 'gelu'
    has_bias: bool = False
    shared_norm: bool = True
    parallel_attention: bool = True

@register_model_config('falcon_40b')
class Falcon40BConfig(Falcon7BConfig):
    decoder_layers: int = 60
    embed_dim: int = 8192
    decoder_ffn_dim: int = 32768
    decoder_attention_heads: int = 128
    attention_heads_kv: int = 8
    shared_norm: bool = False

@register_model_config('adapter_transformer_big')
class AdapterTransformerBigConfig(AdapterTransformerConfig, TransformerBigConfig):
    pass

@register_model_config('adapter_transformer_small')
class AdapterTransformerSmallConfig(AdapterTransformerConfig, TransformerSmallConfig):
    pass

@register_model_config('adapter_transformer_wide')
class AdapterTransformerWideConfig(AdapterTransformerConfig, TransformerWideConfig):
    pass

@register_model_config('hybrid_transformer_big', 'rnmt_big')
class HybridTransformerBigConfig(HybridTransformerConfig):
    embed_dim: int = 1024
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_hidden_size: int = 1024

@register_model_config('hybrid_transformer_wide')
class HybridTransformerWideConfig(HybridTransformerBigConfig):
    encoder_ffn_dim: int = 8192
    decoder_hidden_size: int = 2048

@register_model_config('hybrid_transformer_small')
class HybridTransformerSmallConfig(HybridTransformerConfig):
    embed_dim: int = 512
    encoder_ffn_dim: int = 1024
    decoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 4
    decoder_attention_heads: int = 4
    decoder_hidden_size: int = 512

@register_model_config('adapter_nllb_600m')
class AdapterNLLB600MConfig(AdapterTransformerConfig, NLLB600MConfig):
    pass

@register_model_config('adapter_nllb_1b3')
class AdapterNLLB1B3Config(AdapterTransformerConfig, NLLB1B3Config):
    pass

@register_model_config('adapter_nllb_3b3')
class AdapterNLLB3B3Config(AdapterTransformerConfig, NLLB3B3Config):
    pass

@register_model_config('adapter_mbart_large')
class AdapterMBARTConfig(AdapterTransformerConfig, MBARTConfig):
    pass

@register_model_config('moe_transformer_small')
class MOETransformerSmallConfig(MOETransformerConfig, TransformerSmallConfig):
    pass

@register_model_config('moe_transformer_big')
class MOETransformerBigConfig(MOETransformerConfig, TransformerBigConfig):
    pass

@register_model_config('moe_transformer_wide')
class MOETransformerWideConfig(MOETransformerConfig, TransformerWideConfig):
    pass

@register_model_config('adapter_bloom_1b7')
class AdapterBloom1B7Config(AdapterTransformerConfig, Bloom1B7Config):
    pass

@register_model_config('adapter_bloom_7b')
class AdapterBloom7BConfig(AdapterTransformerConfig, Bloom7BConfig):
    pass

@register_model_config('adapter_llama_7b')
class AdapterLlama7BConfig(AdapterTransformerConfig, Llama7BConfig):
    pass

@register_model_config('adapter_llama_13b')
class AdapterLlama13BConfig(AdapterTransformerConfig, Llama13BConfig):
    pass

@register_model_config('whisper_base')
class WhisperConfig(TransformerConfig):
    encoder_layers: int = 6
    decoder_layers: int = 6
    embed_dim: int = 512
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_prenorm: bool = True
    decoder_prenorm: bool = True
    activation_fn: str = 'gelu'
    encoder_positional_encoding: str = 'learned'
    decoder_positional_encoding: str = 'learned'
    positional_encoding_shift: int = 0
    scale_embed: bool = False
    input_dim: int = 80
    conv_input_dim: int = 80
    conv_channels: int = 512
    conv_kernel_sizes: list[int] = [3, 3]
    conv_strides: list[int] = [1, 2]
    conv_activation: str = 'gelu'
    encoder_max_len: int = 3000
    decoder_max_len: int = 448
    attention_key_bias: bool = False

@register_model_config('whisper_large')
class WhisperLargeConfig(WhisperConfig):
    encoder_layers: int = 32
    decoder_layers: int = 32
    embed_dim: int = 1280
    conv_channels: int = 1280
    encoder_ffn_dim: int = 5120
    decoder_ffn_dim: int = 5120
    encoder_attention_heads: int = 20
    decoder_attention_heads: int = 20

@register_model_config('t5_base')
class T5BaseConfig(TransformerConfig):
    encoder_layers: int = 12
    decoder_layers: int = 12
    encoder_max_len: int = 512
    decoder_max_len: int = 512
    encoder_prenorm: bool = True
    decoder_prenorm: bool = True
    tied_output_projection: bool = False
    embed_dim: int = 768
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 12
    decoder_attention_heads: int = 12
    encoder_positional_encoding: str = 't5'
    decoder_positional_encoding: str = 't5'
    activation_fn: str = 'geglu'
    rms_norm: bool = True
    has_bias: bool = False
    norm_eps: float = 1e-06
    scale_embed: bool = False
    scale_attn: bool = False
    check_inf: bool = True

@register_model_config('t5_large')
class T5LargeConfig(T5BaseConfig):
    encoder_layers: int = 24
    decoder_layers: int = 24
    embed_dim: int = 1024
    encoder_ffn_dim: int = 2816
    decoder_ffn_dim: int = 2816
    encoder_attention_heads: int = 16
    decoder_attention_heads: int = 16
