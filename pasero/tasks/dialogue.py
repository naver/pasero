# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import logging
import regex
import numpy as np
from typing import Any, Union
from pasero import utils
from pasero.config import register_task, DialogueTaskConfig
from pasero.tasks import LanguageModelingTask
from pasero.tokenizers import bos, eos

logger = logging.getLogger('dialogue')


TEMPLATES = {}

def register_chat_template(name: str):
    def wrapper(fn):
        TEMPLATES[name] = fn
        return fn
    return wrapper


@register_task('dialogue')
class DialogueTask(LanguageModelingTask):
    cfg: DialogueTaskConfig

    def __init__(self, data_dir: str, cfg: DialogueTaskConfig):
        cfg.keep_whitespaces = True
        self.chat_template_fn = TEMPLATES.get(cfg.chat_template)
        assert self.chat_template_fn is not None, f"unknown chat template: '{cfg.chat_template}'"
        if cfg.chat_template == 'chatml':
            cfg.stop_sequences.append('<|im_end|>')
        self.system_prompt = cfg.system_prompt
        self.prompt = f'{self.system_prompt}\nUser: ' if self.system_prompt else 'User: '
        super().__init__(data_dir, cfg)
    
    @property
    def task_info(self) -> dict:
        return {
            **super().task_info,
            'prompt': self.prompt,  # used in playground to initialize the chat box with system prompt and user prompt
            'retriever_config': self.cfg.retriever_config,
        }

    @property
    def inference_options(self) -> dict:
        options = {**super().inference_options, 'task': 'dialogue'}
        for name in 'chat_template', 'system_prompt':
            value = getattr(self.cfg, name)
            if value:  # lang_code and domain_tag are False by default, tags is empty by default
                options[name] = value
        return options

    def input_to_sample(self, input: Union[str, list[dict], list[str]], meta: dict = {}) -> dict:
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
        if isinstance(input, list):  # input is already in a list format
            if all(isinstance(content, str) for content in input):
                # input is a list of user/assistant messages 
                target = [
                    {'role': ('user' if i % 2 == 0 else 'assistant'), 'content': content}
                    for i, content in enumerate(input)
                ]
            else:
                # input is already in the right format (list of dicts)
                assert all(isinstance(content, dict) for content in input)
                target = input
            return {'target': target, 'meta': meta}

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
        decoder_input = self.preprocessor.debinarize(sample_bin['decoder_input'])
        # color in green the target words that are part of the prompt (optionally excluded from the training loss,
        # and teacher forced at inference)
        prompt_mask = sample_bin['prompt_mask']
        make_green = lambda s: "\x1b[32;20m" + s + "\x1b[0m"
        decoder_input = [
            make_green(token) if is_prompt else token
            for token, is_prompt in zip(decoder_input, prompt_mask)
        ]
        decoder_input = ' '.join(decoder_input)
        corpus_id = sample_bin['meta']['corpus_id']
        logger.debug(f'{corpus_id} | line example: {decoder_input}')

    def get_reference(self, sample: dict[str, Any]):
        last_turn = sample['target'][-1]
        if last_turn['role'] == 'assistant':
            return last_turn['content']
        else:
            return None

    def preprocess(
        self,
        sample: dict[str, Any],
        truncate: bool = False,
        tokenize: bool = True,
        append_eos: bool = False,
    ) -> dict[str, Any]:
        r"""
        Like LanguageModelingTask, but takes conversations as input and processes them using a chat template, as 
        defined here: https://huggingface.co/docs/transformers/main/chat_templating

        Contrary to other tasks, like language modeling, we don't automatically prepend a BOS token, but let the 
        template define what BOS tokens should be prepended.

        # Example conversation
        
        ```
        sample['target'] = [
            {'role': 'system', 'content': 'You are a helpful assistant'},  # optional
            {'role': 'user', 'content': 'Hello!'},
            {'role': 'assistant', 'content': 'Greetings, how may I help you?'},
            {'role': 'user', 'content': 'How much is 1 + 1?'},
            {'role': 'assistant', 'content': 'Sorry, I cannot answer this question'},
        ]
        ```

        At training, using the "zephyr" format, this would result in:

        ```
        <|system|>\nYou are a helpful assistant</s>\n    # mask=1
        <|user|>\nHello!</s>\n                           # mask=1
        <|assistant|>\n                                  # mask=1
        Greetings, how may I help you?</s>\n             # mask=0
        <|user|>\nHow much is 1 + 1?</s>\n               # mask=1
        <|assistant|>\n                                  # mask=1, end of evaluation prompt
        Sorry, I cannot answer this question</s>\n       # mask=0
        ```

        `mask` is used in the loss computation to exclude the prompt tokens (i.e., a mask of 1 corresponds to a prompt
        token and a mask of 0 to a generated token). At evaluation (with `Trainer.inference_step`), the prompt 
        is made of all the tokens up to the LAST such masked token. This means that only the last assistant's answer 
        is generated and evaluated against.

        At inference, the assistant's answer is stripped of its end tokens (so that the model can continue previously
        started generations):

        ``` 
        <|system|>\nYou are a helpful assistant</s>\n
        <|user|>\nHello!</s>\n
        <|assistant|>\nGreetings, how may I help you?</s>\n
        <|user|>\nHow much is 1 + 1?</s>\n
        <|assistant|>\nSorry, I cannot answer this question
        ```

        And the assistant prompt is added if the last message in the conversation was by the user:

        ```
        <|system|>\nYou are a helpful assistant</s>\n
        <|user|>\nHello!</s>\n
        <|assistant|>\nGreetings, how may I help you?</s>\n
        <|user|>\nHow much is 1 + 1?</s>\n
        <|assistant|>\n
        ```
        """
        assert tokenize, 'dialogue preprocessing is not compatible `tokenize=False`'  # pre-processing needs to apply
        # the chat template and tokenization in order to compute the prompt mask, this is not doable with 
        # pre-templated/pre-tokenized inputs

        conversation = sample['target']
        add_generation_prompt = False

        if not append_eos:
            if not conversation or len(conversation) == 1 and conversation[0]['role'] == 'system':
                conversation.append({'role': 'user', 'content': ''})  # some chatbots behave weirdly when the 
                # conversation is empty: add an empty user message
                add_generation_prompt = True
            elif conversation and conversation[-1]['role'] == 'assistant' and not conversation[-1]['content']:
                # At inference, if the last role is the assistant but its content is empty, remove it but 
                # ensure that assistant start tokens will be added at the end of the sequence. If we kept this empty 
                # message, we would also have assistant end tokens, which would prevent the model from generating the 
                # assistant's answer
                conversation = conversation[:-1]
                add_generation_prompt = True
            elif conversation and conversation[-1]['role'] == 'user':
                # At inference, if the last message is the user's, we don't want to continue it, but to generate the 
                # assistant's message
                add_generation_prompt = True

        # We need to convert the conversation to a single string using the chat template and tokenize it, but 
        # keep track of which parts are the assistant's answers and which parts are user or system prompts.
        decoder_input = []
        prompt_mask = []
        
        formatted = self.chat_template_fn(conversation, add_generation_prompt=add_generation_prompt)
        last_turn = conversation[-1]
        if not append_eos and last_turn['role'] == 'assistant' and last_turn['content']:
            # At inference, strip the end tokens after the assistant message if it is not empty: lets us continue
            # generating from a partial assistant answer
            start = formatted.rfind(last_turn['content'])
            formatted = formatted[:start] + last_turn['content']
        all_tokens = self.preprocessor.tokenize(formatted)
        append_eos = append_eos and eos not in all_tokens  # append EOS only if it not added by the chat template
        decoder_input = self.preprocessor.binarize(all_tokens, append_eos=append_eos)
        prompt_mask = np.ones_like(decoder_input, dtype=bool)

        # We need to tokenize message by message, in order to build the prompt mask (user messages are part of 
        # the prompt while assistant messages are not). However, with some preprocessors: tok(x + y) != tok(x) + tok(y)
        # The only robust way to do that is to do a diff with the tokenization of the previous messages:
        # tok(y) = tok(x + y) - tok(x)
        for i, message in enumerate(conversation):
            if message['role'] != 'assistant':  # by default the prompt is set to True and we want to set it to False
                # at assistant tokens
                continue
            prev = self.chat_template_fn(conversation[:i], add_generation_prompt=True)  # the generation prompt 
            # (e.g., "<|assistant|>") is not part of the assistant message
            current = self.chat_template_fn(conversation[:i + 1], add_generation_prompt=False)
            assert current.startswith(prev)
            prev_tokens = self.preprocessor.tokenize(prev)
            current_tokens = self.preprocessor.tokenize(current)
            assert all_tokens[:len(prev_tokens)] == prev_tokens, (
                'this preprocessor is not compatible with this chat template'
            )
            # this may happen when the preprocessor puts in the same subword characters that 
            # are not part of the same message, for instance the BLOOM preprocessor tokenizes <|im_start|><|im_end|>
            # as [..., '|', '><', '|', ...]
            # The only solution in this case is to use a different chat template
            prev_len = len(prev_tokens)
            current_len = len(current_tokens)
            if i == len(conversation) - 1 and append_eos:  # consider the final EOS as part of the assistant's answer.
                # Not doing that results in 'prompt_len' being inacurate and evaluation at training to use the 
                # wrong prefix
                current_len += 1
            prompt_mask[prev_len:current_len] = False

        if truncate:  # TODO: smarter truncation where earlier dialogue turns are dropped
            decoder_input = decoder_input[:self.max_len]
            prompt_mask = prompt_mask[:self.max_len]
        
        if len(decoder_input) > self.max_len:
            assert not truncate  # this shouldn't happen since we truncate
            return {}
        else:
            return {
                'decoder_input': decoder_input,
                'prompt_mask': prompt_mask,
                'meta': sample['meta'],
            }


@register_chat_template('chatml')
def apply_chatml_template(conversation: list[dict], add_generation_prompt: bool = True) -> str:
    output = []
    for message in conversation:
        role, content = message['role'], message['content']
        output.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
    if add_generation_prompt:
        output.append('<|im_start|>assistant\n')
    return ''.join(output)


@register_chat_template('llama-2')
def apply_llama_template(conversation: list[dict], add_generation_prompt: bool = True) -> str:
    output = []
    if conversation and conversation[0]['role'] == 'system':
        # the system message is merged into the first user message (output conversation won't have a separate system 
        # message)
        system_message = conversation[0]['content']
        system_message = f'<<SYS>>\n{system_message}\n<</SYS>>\n\n'
        conversation = conversation[1:]
    else:
        system_message = ''
    
    for message in conversation:
        role, content = message['role'], message['content']
        if role == 'user':
            output.append(f'{bos}[INST] {system_message}{content.strip()} [/INST] ')
            # FIXME: there should be a whitespace after [/INST], because because Llama-2 chat always generates 
            # an isolated whitespace '▁' after the instruction, which suggests it was trained this way. For instance: 
            # "<s> ▁[ INST ] ▁ 1 + 1 = ? ▁Please ▁answer ▁with ▁digits ▁only ▁[ / INST ]" ->
            # "▁ ▁Sure ! ▁ 1 ▁+ ▁ 1 ▁= ▁ 2 . </s>"
            # However, with the current preprocessing: the whitepace after [/INST] and the whitespace before 
            # the assistant messages in the prompt would be merged into a single "▁▁" token, which Llama-2 doesn't 
            # like!
            # One dirty solution would be to filter the tokenizer's output to replace
            # "▁[ / INST ] ▁▁" with "▁[ / INST ] ▁".
            system_message = ''
        elif role == 'assistant':
            output.append(f' {content.strip()} {eos}')
        else:
            raise Exception
    return ''.join(output)


@register_chat_template('mistral')
def apply_mistral_template(conversation: list[dict], add_generation_prompt: bool = True) -> str:
    """
    Like llama-2, except that it does not accept a system prompt.
    """
    output = []
    if conversation and conversation[0]['role'] == 'system':
        if conversation[0]['content']:
            utils.warn_once(logger, 'the mistral chat template does not support system prompts, those will be ignored')
        conversation = conversation[1:]
    for i, message in enumerate(conversation):
        role, content = message['role'], message['content']
        prefix = bos if i == 0 else ''
        if role == 'user':
            output.append(f'{prefix}[INST] {content} [/INST]')   # TODO: should we strip?
        elif role == 'assistant':
            output.append(f' {content}{eos}')
        else:
            raise Exception
    return ''.join(output)


@register_chat_template('zephyr')
def apply_zephyr_template(conversation: list[dict], add_generation_prompt: bool = True) -> str:
    output = []
    for message in conversation:
        role, content = message['role'], message['content']
        output.append(f'<|{role}|>\n{content}{eos}\n')
    if add_generation_prompt:
        output.append('<|assistant|>\n')
    return ''.join(output)


@register_chat_template('solar')
def apply_solar_template(conversation: list[dict], add_generation_prompt: bool = True) -> str:
    output = []
    for message in conversation:
        role, content = message['role'], message['content']
        if role == 'system':
            output.append(f'### System:\n{content}\n\n')
        elif role == 'user':
            output.append(f'### User:\n{content}\n\n')
        elif role == 'assistant':
            output.append(f'### Assistant:\n{content}\n\n')  # contrary to the official template, we add \n\n to support 
            # multi-turn conversations. In single-turn conversations, this shouldn't change anything.
            # FIXME: there is no EOS, even though the official model seems to generate one after the assistant answers.
            # This doesn't seem to be a problem at inference. But if we use this template at training, there won't be 
            # a clear delimiter between turns. We cannot add EOS at every turn because the SOLAR model seems to 
            # ignore everything before EOS.
    if add_generation_prompt:
        output.append('### Assistant:\n')
    return ''.join(output)
