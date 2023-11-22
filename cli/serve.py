#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import sys
import logging
import argparse
import time
import itertools
import functools
import json
from numbers import Number
from torch import multiprocessing as mp
from flask import Flask, jsonify, request, render_template, Response
from waitress import serve

app = Flask(__name__, template_folder='.')

from pasero import utils
from pasero.decoding import TextGenerator


parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+', help='paths of the models to serve (model directory or checkpoint). Each '
                    'model can be followed by a device id following this format: "PATH:GPU_ID"', default=[])
parser.add_argument('--port', type=int, default=8000, help='listen for HTTP on this port')
parser.add_argument('--dtype', help="override the models' default type with this one")
utils.init_logging(stream=sys.stderr)
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
    'source_lang': (str, 'source language'),
    'target_lang': (str, 'target language'),
    'stop_regex': (str, 'regular-expression that will stop generation when matched'),
}


models = {}


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


def parse_device(device: str) -> str:
    if device == 'cpu':
        return device
    elif device == 'cuda':
        return 'cuda:0'
    device = int(device.removeprefix('cuda:'))
    return f'cuda:{device}'


def parse_devices(devices: list[str]) -> list[str]:
    if not devices:
        return ['cuda:0']
    return [parse_device(device) for device in devices.split(',')]


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


def set_default_langs(model_info: dict) -> None:
    decoding_opts = model_info.setdefault('decoding_options', {})

    default_source_lang = model_info.get('default_source_lang')
    source_langs = model_info.get('source_langs') or []
    for lang in default_source_lang, 'fr', 'fra_Latn':
        if lang and lang in source_langs:
            decoding_opts['source_lang'] = lang
            break
 
    default_target_lang = model_info.get('default_target_lang')
    target_langs = model_info.get('target_langs') or []
    for lang in default_target_lang, 'en', 'eng_Latn':
        if lang and lang in target_langs:
            decoding_opts['target_lang'] = lang
            break


@app.route('/<name>/api', methods=['GET', 'POST'])
def api(name: str):
    if name not in models:
        return Response(status=204)
    
    model_info: dict = models[name].model_info
    set_default_langs(model_info)
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
    model_info = dict(models[name].model_info)
    set_default_langs(model_info)
    return jsonify(model_info)


@app.route('/list', methods=['GET', 'POST'])
def list_models():
    return jsonify([model.model_info for model in models.values()])


@app.route('/help', methods=['GET', 'POST'])
def help():
    return jsonify({'actions': actions, 'options': {name: desc for name, (_, desc) in options.items()}})


def chat(model: TextGenerator, input: str, config: dict):
    if input.rfind('\nUser:') >= input.rfind('\nAssistant:'):
        yield {'detok': '\nAssistant: '}
        input = input + '\nAssistant:'
    for out in model.stream(input, **config):
        yield out
    yield {'detok': '\nUser: '}
    yield {}


def round_all(object, ndigits=5):
    return utils.apply(
        functools.partial(round, ndigits=ndigits),
        object,
        Number,
    )


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
    chat_mode = model.model_info['task'] == 'dialogue'
    if chat_mode:
        output = chat(model, input, config)
    else:
        output = model.stream(input, **config)
        output = itertools.chain(output, [{}])
    
    output = (f'data: {json.dumps(round_all(out))}\n\n' for out in output)
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

    # Only get the top-1 hypotheses
    output = [nbest[0] for nbest in output]
    # Convert a list of dicts into a dict of list
    # While the list format is convenient, we need to send other things that are not lists (e.g., elapsed time)
    if output:
        output = {
            k: [hyp[k] for hyp in output]
            for k in output[0]
        }
    output = utils.array_to_list(output)
    output.update({
        'elapsed': elapsed,
        'model': name,
    })
    output['detok'] = [
        line.strip().removesuffix('\\n').replace('\\n', '\n')
        for line in output['detok']
    ]
    output['input'] = input  # should this be normalized?

    output = round_all(output)
    logger.info('output: %s', output)
    return jsonify(output)



def main():
    args = parser.parse_args()
    
    mp.set_start_method("spawn")

    for model_path in args.models:
        model_path = model_path.rstrip('/')
        model_path, *device = model_path.rsplit(':', maxsplit=1)
        opts = {}
        opts['devices'] = parse_devices(device[0] if device else None)
        if args.dtype:
            opts['dtype'] = args.dtype
        model = TextGenerator.build(model_path, **opts)
        model_name = model.model_info['name']
        models[model_name] = model
    
    serve(app, host="0.0.0.0", port=args.port)

if __name__ == '__main__':
    main()
