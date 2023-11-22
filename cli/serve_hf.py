#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import sys
import re
import logging
import argparse
import itertools
import time
import json
import torch
from queue import Queue
from threading import Event
from typing import Optional
from torch import multiprocessing as mp
from flask import Flask, jsonify, request, render_template, Response
from waitress import serve
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer


app = Flask(__name__, template_folder='.')


parser = argparse.ArgumentParser(description="Like pasero-serve, but can be used to serve HuggingFace models")
parser.add_argument('models', nargs='+', help='paths of the models to serve (model directory or checkpoint). Each '
                    'model can be followed by a device id following this format: "PATH:GPU_ID"', default=[])
parser.add_argument('--port', type=int, default=8000, help='listen for HTTP on this port')
parser.add_argument('--dtype', help="override the models' default type with this one")


logging.basicConfig(
    format='%(asctime)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level='INFO',
    stream=sys.stderr,
)

logger = logging.getLogger('server')


actions = {
    '/help': 'get a list of available actions and options with their description',
    '/list': 'get a list of all available models',
    '<model>/decode': 'do batched decoding given inputs with given model',
    '<model>/stream': 'stream tokens with given model from given input',
    '<model>/info': 'get information about given model (task, covered languages, etc.)',
}

options = {
    'input': (list, 'text input to translate or use as prompt'),
    'beam_size': (int, 'beam size in beam search decoding (default: 5)'),
    'sampling': (bool, 'use sampling instead of greedy decoding'),
    'sampling_topp': (float, 'do nucleus sampling with this top-p value'),
    'sampling_topk': (int, 'do top-k sampling with this k value'),
    'sampling_temperature': (float, 'do sampling with this softmax temperature'),
    'repeat_penalty': (float, 'penalty for repeated tokens (default: 1)'),
    'max_output_len': (int, 'maximum number of tokens to generate'),
    'stop_regex': (str, 'regular-expression that will stop generation when matched'),
}


models = {}


class TokenStreamer:
    """
    Improved version of HuggingFace transformers' TextIteratorStreamer, which yields the detokenized words as well
    as the corresponding tokens.

    Also has a `stop` method to interrupt generation and free memory (called when a client disconnects in the middle 
    of generation)
    """
    def __init__(self, tokenizer: AutoTokenizer):
        self.token_queue = Queue()
        self.stopped = Event()
        self.tokenizer = tokenizer

    def stop(self):
        self.stopped.set()

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        if self.stopped.is_set():
            raise Exception('generation interrupted by client')   # interrupts `.generate()` to free memory
        assert len(value.shape) == 1 or value.shape[0] == 1
        self.token_queue.put(value[0])

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        self.token_queue.put(None)

    def detokenize(self, tokens: list[str]):
        tokens = [token for token in tokens if token not in self.tokenizer.all_special_tokens]
        return self.tokenizer.convert_tokens_to_string(tokens)

    def __iter__(self):
        is_prompt = True
        all_tokens = []
        
        while True:
            value = self.token_queue.get()
            if value is None:
                break
            if len(value.shape) == 0:
                value = [value]
            tokens = self.tokenizer.convert_ids_to_tokens(value)

            if is_prompt:
                yield None, tokens
                is_prompt = False
                all_tokens += tokens
                prev_detok = self.detokenize(all_tokens)
                continue

            for token in tokens:
                all_tokens.append(token)
                detok = self.detokenize(all_tokens).rstrip('�')
                word = detok[len(prev_detok):]
                yield word, [token]
                prev_detok = detok


class HuggingFaceModel:
    def __init__(
        self,
        model_path,
        device: str = 'cuda:0',
        dtype: Optional[str] = None,
    ):  
        self.name = os.path.basename(model_path)
        
        dtype = dtype or 'float16'
        dtype = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device,
        )
        self.max_len = self.model.config.max_position_embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            truncation_side='left',
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.is_chat_model = self.tokenizer.chat_template is not None
        
        system_prompt = 'You are a helpful, respectful and honest AI assistant.'
        if not self.is_chat_model:
            self.prompt = None
        elif system_prompt:
            self.prompt = f'{system_prompt}\nUser: '
        else:
            self.prompt = f'User: '

        self.model_info = {
            'name': self.name,
            'max_len': self.max_len,
            'prompt': self.prompt,
            'model_type': 'decoder',
            'task': 'dialogue' if self.is_chat_model else 'language_modeling',
            'decoding_options': {'sampling': True},
        }

    def input_to_conversation(self, input: str) -> list[dict]:
        pattern = r'(\nUser:|\nAssistant:)'
        if not re.search(pattern, '\n' + input):
            input = f'User: {input}'  # for interactive use with pasero-decode
        raw_conversation = re.split(pattern, '\n' + input)
        
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
        return conversation

    def preprocess(self, inputs: list[str]):
        if self.is_chat_model:
            inputs_new = []
            for input in inputs:
                conversation = self.input_to_conversation(input)
                
                if conversation and conversation[-1]['role'] == 'assistant':
                    suffix = conversation.pop()['content']  # to prevent `apply_chat_template` from appending 
                    # end of message tokens after the assistant's message (we want to continue it)
                else:
                    suffix = None
                input = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if suffix is not None:
                    input += suffix
                inputs_new.append(input)
            inputs = inputs_new

        input_tok = self.tokenizer(
            inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,  # TODO: smarter truncation that keeps the system message and correct format
            max_length=self.max_len - 10,
        ).to(self.model.device)
        input_tok.pop('token_type_ids', None)  # incompatible with LLaMA
        return input_tok

    def stream(
        self,
        input: str,
        max_output_len: int = 100,
        beam_size: int = 1,
        sampling: bool = True,
        sampling_temperature: float = 1.0,
        sampling_topp: float = 1.0,
        sampling_topk: Optional[float] = None,
        repeat_penalty: float = 1.0,
        stop_regex: Optional[str] = None,
        **kwargs,
    ):
        input_tok = self.preprocess([input])
        prompt_len = input_tok.input_ids.size(1)
        max_output_len = min(self.max_len - prompt_len, max_output_len)

        if sampling_temperature == 0 or beam_size > 1:
            sampling = False
        
        streamer = TokenStreamer(self.tokenizer)

        kwargs = dict(
            input_tok,
            streamer=streamer,
            max_new_tokens=max_output_len,
            num_beams=beam_size,
            do_sample=sampling,
            temperature=sampling_temperature,
            top_p=sampling_topp,
            sampling_topk=sampling_topk,
            repetition_penalty=repeat_penalty,
        )

        try:
            thread = Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()

            if self.is_chat_model:
                if input.rfind('\nUser:') >= input.rfind('\nAssistant:'):
                    yield {'detok': '\nAssistant: '}

            text = ''
            tokens = []
            stream = iter(streamer)
            while True:
                try:
                    start = time.time()
                    word, tokens = next(stream)
                    elapsed = time.time() - start
                except StopIteration:
                    break

                if word is None:
                    yield {'prompt_tokens': tokens, 'elapsed': elapsed}
                    tokens = []
                else:
                    text += word
                    yield {'tokens': tokens, 'detok': word, 'elapsed': elapsed}
                    if stop_regex and re.search(stop_regex, text):
                        break

            if self.is_chat_model:
                yield {'detok': '\nUser: '}
        except GeneratorExit:  # happens when a client disconnects
            streamer.stop()

    def decode(
        self,
        *inputs: str,
        max_output_len: int = 100,
        beam_size: int = 1,
        sampling: bool = True,
        sampling_temperature: float = 1.0,
        sampling_topp: float = 1.0,
        sampling_topk: Optional[float] = None,
        repeat_penalty: float = 1.0,
        **kwargs,
    ) -> list[str]:
        input_tok = self.preprocess(inputs)
        prompt_len = input_tok.input_ids.size(1)
        max_output_len = min(self.max_len - prompt_len, max_output_len)

        if sampling_temperature == 0 or beam_size > 1:
            sampling = False
        
        kwargs = dict(
            input_tok,
            max_new_tokens=max_output_len,
            num_beams=beam_size,
            do_sample=sampling,
            temperature=sampling_temperature,
            top_p=sampling_topp,
            sampling_topk=sampling_topk,
            repetition_penalty=repeat_penalty,
        )

        out = self.model.generate(**kwargs)
        out = out[:,prompt_len:]
        detok = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        if self.is_chat_model:
            for i, input in enumerate(inputs):
                if input.rfind('\nUser:') >= input.rfind('\nAssistant:'):
                    detok[i] = '\nAssistant: ' + {detok[i]}
        return detok


def process_input(input: list[str]) -> list[str]:
    if not input:
        return []
    else:
        return [line.replace('\r', '') for line in input if line.strip()]


class BadRequest(Exception):
    status = 400


def get_model(name):
    model = models.get(name)
    if model is None:
        raise BadRequest(f'model {name} does not exist')
    return model


def strtobool(val: str) -> bool:
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid boolean value '{val}'")


def get_data():
    if request.method == 'GET':
        data = request.args
    else:
        data = request.form

    input = data.getlist('input[]') or data.getlist('input')
    data = data.to_dict()
    data.pop('input[]', None)
    data.pop('input', None)
    if input:
        data['input'] = input

    clean_data = {}
    
    for name, value in data.items():
        if name in options:
            type, _ = options[name]
            if type in (int, float, str, list):
                value = type(value)
            elif type == bool:
                value = strtobool(value)
            else:
                raise NotImplementedError
            clean_data[name] = value
        else:
            raise ValueError(f"unknown option '{name}'")
    
    data = clean_data

    data['max_output_len'] = max(1, data.get('max_output_len', 100))
    return data


def parse_device(device: Optional[str]) -> str:
    if device == 'cpu' or device == 'auto':
        return device
    elif device == 'cuda' or device is None:
        return 'cuda:0'
    device = int(device.removeprefix('cuda:'))
    return f'cuda:{device}'


@app.route('/<name>', methods=['GET', 'POST'])
def playground(name: str):
    if name not in models:
        return Response(status=204)
    return render_template('playground.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template(
        'index.html',
        models=[model.model_info for model in models.values()],
    )


@app.route('/<name>/api', methods=['GET', 'POST'])
def api(name: str):
    if name not in models:
        return Response(status=204)
    
    model_info: dict = models[name].model_info
    decoding_opts = model_info.get('decoding_options', {})
    decoding_opts_repr = {repr(k): repr(v) for k, v in decoding_opts.items()}
    model_info['decoding_options'] = decoding_opts_repr  # for pretty printing
    
    return render_template(
        'api.html',
        model_info=model_info,
    )


@app.route('/<name>/info', methods=['GET', 'POST'])
def model_info(name: str):
    if name not in models:
        return Response(status=204)
    model_info: dict = models[name].model_info
    return jsonify(model_info)


@app.route('/list', methods=['GET', 'POST'])
def list_models():
    return jsonify([model.model_info for model in models.values()])


@app.route('/help', methods=['GET', 'POST'])
def help():
    return jsonify({'actions': actions, 'options': {name: desc for name, (_, desc) in options.items()}})


@app.route('/<name>/stream', methods=['GET'])
def stream(name):
    config = get_data()
    config['beam_size'] = 1
    logger.info('input: %s', config)
    model = get_model(name)
    input = process_input(config.pop('input', None))
    if not input:
        input = ['']
    assert len(input) == 1
    # normalize the prompt to remove unnecessary whitespaces
    input = input[0]
    logger.warning(input.encode())
    output = model.stream(input, **config)
    output = itertools.chain(output, [{}])
    output = (f'data: {json.dumps(out)}\n\n' for out in output)
    return Response(output, mimetype='text/event-stream')


@app.route('/<name>/decode', methods=['GET', 'POST'])
def decode(name):
    config = get_data()
    logger.info('input: %s', config)
    model = get_model(name)
    input = process_input(config.pop('input', None))
    start = time.time()
    output = model.decode(*input, **config, return_scores=True)
    elapsed = time.time() - start
    output = [
        line.strip().removesuffix('\\n').replace('\\n', '\n')
        for line in output
    ]
    output = {
        'elapsed': elapsed,
        'detok': output,
    }
    logger.info('output: %s', output)
    return jsonify(output)


def main():
    args = parser.parse_args()
    mp.set_start_method("spawn")

    for model_path in args.models:
        model_path = model_path.rstrip('/')
        model_path, *device = model_path.rsplit(':', maxsplit=1)
        device = parse_device(device[0] if device else None)
        model = HuggingFaceModel(model_path, device=device, dtype=args.dtype)
        model_name = model.model_info['name']
        models[model_name] = model
    
    serve(app, host="0.0.0.0", port=args.port)


if __name__ == '__main__':
    main()
