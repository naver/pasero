# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import logging
import regex
import numpy as np
from typing import Any, Union
from pasero.config import DialogueTaskConfig
from pasero.tasks import LanguageModelingTask, MonolingualCorpus

logger = logging.getLogger('language_modeling')


TEMPLATES = {
    # https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca/blob/main/tokenizer_config.json
    'chatml': "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endfor %}"
              "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",  # modified to remove the newline after <|im_end|>, which is wrongly tokenized
    'llama-2': "{% if messages and messages[0]['role'] == 'system' %}{% set system_message = '<<SYS>\n' + messages.pop(0)['content'] + '\n<<SYS>\n\n' %}{% endif %}"
               "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<s>[INST] ' }}{% if system_message is defined and loop.index == 1 %}{{ system_message }}{% endif %}{{ message['content'] + ' [/INST]' }}{% else %}{{ message['content'] + ' </s>' }}{% endif %}{% endfor %}",
    # FIXME: --tokenizer sentencepiece doesn't work with this format, as it tokenizes <s> and </s>
    'zephyr': "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>' }}{% elif message['role'] == 'user' %}{{ '<|user|>' }}{% else %}{{ '<|assistant|>' }}"
              "{% endif %}{{ '\n' + message['content'] + '</s>' + '\n'}}{% endfor %}"
              "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}",
}
# TODO: check possible whitespace discrepancies between training and inference


class DialogueTask(LanguageModelingTask):
    cfg: DialogueTaskConfig

    def __init__(self, data_dir: str, cfg: DialogueTaskConfig):
        cfg.keep_whitespaces = True
        super().__init__(data_dir, cfg)
        self.chat_template = cfg.custom_chat_template or TEMPLATES.get(cfg.chat_template)
        assert self.chat_template is not None
        if cfg.chat_template == 'chatml':
            stop_token = '<|im_end|>'
            if stop_token not in cfg.stop_sequences:
                cfg.stop_sequences.append(stop_token)
        self.system_prompt = cfg.system_prompt
        self.prompt = f'{self.system_prompt}\nUser: ' if self.system_prompt else 'User: '
    
    @property
    def task_info(self) -> dict:
        return {
            **super().task_info,
            'prompt': self.prompt,  # used in playground to initialize the chat box with system prompt and user prompt
        }

    @property
    def inference_options(self) -> dict:
        options = {**super().inference_options, 'task': 'dialogue'}
        for name in 'chat_template', 'custom_chat_template', 'system_prompt':
            value = getattr(self.cfg, name)
            if value:  # lang_code and domain_tag are False by default, tags is empty by default
                options[name] = value
        return options

    def input_to_sample(self, input: Union[str, list[dict]], meta: dict) -> dict:
        """
        Parse conversations of this format:

        You are a helpful assistant that is very good at math.
        User: Hello, do you know how much is 1+1?
        Assistant: Sure, it is 3!

        The example above is parsed as:
        [
            {'role': 'system', 'content': 'You are a helpful assistant that is very good at math.'},
            {'role': 'user', 'content': 'Hello, do you know how much is 1+1?'},
            {'role': 'assistant', 'content': 'Sure, it is 3!'},
        ]
        """
        if isinstance(input, list):  # input is already in the right format
            assert all(isinstance(turn, dict) for turn in input)
            return {'target': input, 'meta': meta}
        
        pattern = r'(\nUser:|\nAssistant:)'
        if not regex.search(pattern, '\n' + input):
            input = f'User: {input}'  # for interactive use with pasero-decode
        raw_conversation = regex.split(pattern, '\n' + input)
        
        role = 'system'
        conversation = []
        for content in raw_conversation:
            if content == '\nUser:':
                role = 'user'
            elif content == '\nAssistant:':
                role = 'assistant'
            else:
                content = content.strip()  # remove extra whitespaces around the "User:" / "Assistant:" delimiters
                if content or role != 'system':  # do not add system if its content is empty, but empty assistant 
                    # or user messages are allowed
                    conversation.append({'role': role, 'content': content})
        
        if self.system_prompt and conversation and conversation[0]['role'] != 'system':
            # add system prompt if --system-prompt is set and input doesn't have one
            conversation.insert(0, {'role': 'system', 'content': self.system_prompt})
        return {'target': conversation, 'meta': meta}

    def log_sample(self, sample_bin: dict) -> None:
        target_tok = self.tgt_preprocessor.debinarize(sample_bin['target'], keep_padding=True)
        prompt_mask = sample_bin['prompt_mask']
        make_green = lambda s: "\x1b[32;20m" + s + "\x1b[0m"
        tokens = target_tok.split()
        tokens = [make_green(token) if is_prompt else token for token, is_prompt in zip(tokens, prompt_mask)]
        target_tok = ' '.join(tokens)
        corpus_id = sample_bin['meta']['corpus_id']
        logger.debug(f'{corpus_id} | line example: {target_tok}')

    def preprocess(
        self,
        sample: dict[str, Any],
        truncate: bool = False,
        tokenize: bool = True,
        inference: bool = False,
    ) -> dict[str, Any]:
        """
        Like LanguageModelingTask, but takes conversations as input and processes them using a chat template, as 
        defined here: https://huggingface.co/docs/transformers/main/chat_templating

        # Example conversation
        
        sample['target'] = [
            {'role': 'system', 'content': 'You are a helpful assistant'},  # optional
            {'role': 'user', 'content': 'Hello!'},
            {'role': 'assistant', 'content': 'Greetings, how may I help you?'},
            {'role': 'user', 'content': 'How much is 1 + 1?'},
            {'role': 'assistant', 'content': 'Sorry, I cannot answer this question'},
        ]

        At training, using the "zephyr" format, this would result in:

            <|system|>\nYou are a helpful assistant</s>\n                 # mask=1
            <|user|>\nHello!</s>\n                                        # mask=1
            <|assistant|>\nGreetings, how may I help you?</s>\n           # mask=0
            <|user|>\nHow much is 1 + 1?</s>\n                            # mask=1, end of evaluation prompt
            <|assistant|>\nSorry, I cannot answer this question</s>\n     # mask=0

        `mask` is used in the loss computation to exclude the prompt tokens

        At inference, the assistant's answer is stripped of its end tokens (so that the model can continue previously
        started generations):
            
            <|system|>\nYou are a helpful assistant</s>\n
            <|user|>\nHello!</s>\n
            <|assistant|>\nGreetings, how may I help you?</s>\n
            <|user|>\nHow much is 1 + 1?</s>\n
            <|assistant|>\nSorry, I cannot answer this question

        And the assistant prompt is added if the last message in the conversation was by the user:

            <|system|>\nYou are a helpful assistant</s>\n
            <|user|>\nHello!</s>\n
            <|assistant|>\nGreetings, how may I help you?</s>\n
            <|user|>\nHow much is 1 + 1?</s>\n
            <|assistant|>\n

        """
        # Jinja2 templating copied from HuggingFace Transformers:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1781
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        compiled_template = jinja_env.from_string(self.chat_template)

        conversation = sample['target']
        add_generation_prompt = False

        if inference and (not conversation or len(conversation) == 1 and conversation[0]['role'] == 'system'):
            conversation.append({'role': 'user', 'content': ''})  # some chatbots behave weirdly when the conversation 
            # is empty: add an empty user message
            add_generation_prompt = True
        elif (
            inference and
            conversation and conversation[-1]['role'] == 'assistant' and not conversation[-1]['content']
        ):
            # At inference, if the last role is the assistant but its content is empty, remove it but 
            # ensure that assistant start tokens will be added at the end of the sequence. If we kept this empty 
            # message, we would also have assistant end tokens, which would prevent the model from generating the 
            # assistant's answer
            conversation = conversation[:-1]
            add_generation_prompt = True
        elif inference and conversation and conversation[-1]['role'] == 'user':
            add_generation_prompt = True

        # We need to convert the conversation to a single string using the chat template and tokenize it, but 
        # keep track of which parts are the assistant's answers and which parts are user or system prompts.
        target_bin = []
        target_mask = []
        prev_target_tok = ''
        
        for i in range(len(conversation)):
            role = conversation[i]['role']
            content = conversation[i]['content']
            is_last_turn = (i == len(conversation) - 1)

            # Tokenize the entire conversation up to this turn
            target = compiled_template.render(
                messages=conversation[:i + 1],
                add_generation_prompt=add_generation_prompt and is_last_turn,
            )

            if inference and is_last_turn and role == 'assistant' and content:
                # At inference, strip the end tokens after the assistant message if it is not empty: lets us continue
                # generating from a partial assistant answer
                start = target.rfind(content)
                target = target[:start] + content
            
            target_tok = self.tgt_preprocessor.tokenize(target) if tokenize else target
            target_tok = target_tok.removeprefix('<s>').lstrip(' ')  # the llama-2 template prepends a BOS token
            # to each turn, but `collate` also prepends this token when building the decoder input
            
            # Then find this segment's tokens by stripping the tokens generated at the previous turns
            segment_tok = target_tok.removeprefix(prev_target_tok).lstrip(' ')
            append_eos = inference and is_last_turn and not segment_tok.endswith('</s>')
            segment_bin = self.tgt_preprocessor.binarize(segment_tok, append_eos=append_eos)
            is_prompt = role != 'assistant'
            segment_mask = np.full_like(segment_bin, fill_value=is_prompt, dtype=bool)

            target_bin.append(segment_bin)
            target_mask.append(segment_mask)
            prev_target_tok = target_tok

        target_bin = np.concatenate(target_bin)
        target_mask = np.concatenate(target_mask)

        assert len(target_bin) == len(target_mask)

        if truncate:
            target_bin = target_bin[:self.max_len]
            target_mask = target_mask[:self.max_len]
        
        if len(target_bin) > self.max_len:
            assert not truncate  # this shouldn't happen since we truncate
            return {}
        else:
            return {
                'target': target_bin,
                'prompt_mask': target_mask,
                'meta': sample['meta'],
            }
    
    @classmethod
    def _get_corpus(cls, *args, **kwargs) -> MonolingualCorpus:
        corpus = super()._get_corpus(*args, **kwargs)
        corpus.file_formats = ['jsonl']
        return corpus

    @classmethod
    def get_inference_corpus(cls, *args, **kwargs) -> MonolingualCorpus:
        corpus = super().get_inference_corpus(*args, **kwargs)
        corpus.file_formats = ['jsonl']
        return corpus
