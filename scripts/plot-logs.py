#!/usr/bin/env python
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import os
import subprocess
import tempfile
from matplotlib import pyplot as plt
from pasero.utils import parse_logs

description = """
Plot validation or training metrics for a list of models by parsing their training logs.

Example of usage:
```
scripts/plot-logs.py FLORES-valid.fr-en --models models/ParaCrawl/fr-en.{big.6-6,base.6-6} --metric bleu -o bleu.pdf \
--xlabel "Training steps" --ylabel "French-English validation BLEU" --model-names "Transformer Big" "Transformer Base"
# Plots the BLEU validation curves of two models into the file "bleu.pdf"
```

This can also be used remotely via thanks to the `--host-name` option. For instance:
```
scripts/plot-logs.py train --models fr-en.{big.6-6,base.6-6} --host-name SOME_SERVER_NAME --root-dir models/ParaCrawl \
--metric ppl
# Plots the perplexity training curves of two remote models to the screen
```
Note that `SOME_SERVER_NAME` should be a server that is accessible through SSH.
"""

METRICS = ['bleu', 'chrf', 'ppl', 'loss', 'nll_loss', 'loss_scale', 'wpb', 'chrf++', 'spbleu', 'gnorm']

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=description)

# Main options
parser.add_argument('train_or_valid_sets', nargs='+', help="name of the validation set in the training logs "
                    "(e.g., 'valid.de-en'), set to 'train' to plot training metrics. If several valid sets are "
                    "provided, their metrics are averaged")
parser.add_argument('--models', required=True, nargs='+', help="list of training log files or model directories "
                    "containing a 'train.log' file")
parser.add_argument('--metric', choices=METRICS, default='nll_loss', help='which metric to plot')
parser.add_argument('--model-names', nargs='+', help="list of pretty model names (default: model directories)")
parser.add_argument('-o', '--output', help="save the figure into this file (default: displayed interactively), "
                    "file extension determines the format (e.g., .pdf or .png)")
parser.add_argument('--aliases', nargs='+', default=[], help="other names the validation set can have in the training "
                    " logs (e.g., 'valid', 'valid.de-en', etc.)")

# Remote usage
parser.add_argument('--host', help='fetch log files on this remote host via SSH')
parser.add_argument('--root-dir', help='log file paths are relative to this directory')

# Figure caption and labels
parser.add_argument('--title', help='title of the figure')
parser.add_argument('--xlabel', help='label for the x-axis', default='Training steps')
parser.add_argument('--ylabel', help='label for the y-axis (default: METRIC)')

# Baseline
parser.add_argument('--baseline', type=float, help="baseline score that will be shown as an horizontal line")
parser.add_argument('--baseline-name', default='baseline', help='name of the baseline')

# Curve styles: telling curves apart
parser.add_argument('--sort', action='store_true', help='sort models in the legend by max score')
parser.add_argument('--markers', action='store_true', help='show all individual points with a marker')
parser.add_argument('--linestyles', nargs='+', choices=['solid', 'dotted', 'dashed', 'dashdot'],
                    help="style of each model's curve")
parser.add_argument('--colors', nargs='+', help="color of each model's curve")

# Position and granularity of the axes
parser.add_argument('--offset', type=int, nargs='+',
                    help='list of step offsets: used to "shift" the scores of a model to the right')
parser.add_argument('--max-steps', type=int, help='only scores for steps under this value will be plotted')
parser.add_argument('--min-steps', type=int, help='only scores for steps above this value will be plotted')
parser.add_argument('--max-points', type=int, default=100, help='maximum number of points per curve')
parser.add_argument('--min-y', type=float, help='minimum value for the Y axis')
parser.add_argument('--max-y', type=float, help='maximum value for the Y axis')
parser.add_argument('--start-at-zero', action='store_true',
                    help='make all model curves start at zero (subtract step value by min step value)')
parser.add_argument('--by-line-count', action='store_true',
                    help='the X axis will be the number of lines instead of the number of batches')

# Figure size and position
parser.add_argument('--fig-size', type=float, nargs=2, help='size of the figure (width, height)')
parser.add_argument('--legend-loc', default='best', help='position of the legend in the plot')
parser.add_argument('--xticks', type=float, nargs='+',
                    help='manually define the values that will be displayed under the X axis')
parser.add_argument('--margin', type=float, nargs=4, help='left/bottom/right/top margins (between 0 and 1)')


all_linestyles = dict([
     ('solid', 'solid'),
     ('dotted', 'dotted'),
     ('dashed', 'dashed'),
     ('dashdot', 'dashdot'),
])

args = parser.parse_args()

if args.ylabel is None:
    args.ylabel = args.metric.upper()

if args.fig_size:
    plt.rcParams["figure.figsize"] = args.fig_size

log_paths = args.models
if args.root_dir:
    log_paths = [os.path.join(args.root_dir, name) for name in log_paths]
log_paths = [name if name.endswith('.log') else os.path.join(name, 'train.log') for name in log_paths]

if args.model_names:
    model_names = args.model_names
else:
    prefix = args.root_dir or ''
    model_names = [os.path.dirname(path).removeprefix(prefix).strip('/') for path in log_paths]

local_log_paths = []
if args.host:
    for path in log_paths:
        out = subprocess.check_output(
            ['ssh', args.host, 'grep', 'nll_loss', path],
            stderr=subprocess.DEVNULL
        ).decode()
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(out)
            local_log_paths.append(temp_file.name)
    log_paths = local_log_paths

assert len(model_names) == len(log_paths)

offsets = args.offset or [0] * len(log_paths)
assert len(offsets) == len(log_paths)

colors = args.colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = [all_linestyles[s] for s in args.linestyles] if args.linestyles else ['solid']
min_x = float('inf')
max_x = float('-inf')
plots = []

line_count = 0
for i, (log_path, model_name, offset) in enumerate(zip(log_paths, model_names, offsets)):
    values = {}
    last_step = None

    all_logs = parse_logs(log_path)

    for subset, logs in all_logs.items():
        if subset in args.aliases and subset not in args.train_or_valid_sets:
            subset = args.train_or_valid_sets[0]  # aliases are for the first valid set

        if subset in args.train_or_valid_sets:
            for steps, metrics in logs.items():
                if args.metric not in metrics:
                    continue
                
                last_step = steps
                min_x = min(min_x, steps)
                max_x = max(max_x, steps)
                values.setdefault(steps, {})[set] = metrics[args.metric]

    if values:
        assert len(set([len(scores) for steps, scores in values.items()])) == 1, 'cannot average metrics over valid ' \
            'sets with a different number of data points'
        values = sorted([(steps, sum(scores.values()) / len(scores)) for steps, scores in values.items()])

        x, y = zip(*values)
        if args.start_at_zero:
            min_ = min(x)
            x = [x_ - min_ for x_ in x]
        
        if args.min_steps:
            x, y = zip(*[(x_, y_) for x_, y_ in zip(x, y) if x_ >= args.min_steps])
        if args.max_steps:
            x, y = zip(*[(x_, y_) for x_, y_ in zip(x, y) if x_ <= args.max_steps])
        
        if args.max_points and len(x) > args.max_points:
            k = len(x) // args.max_points
            x = x[::k]
            y = y[::k]

        plots.append({
            'x': x,
            'y': y,
            'label': model_name,
            'color': colors[i % len(colors)],
            'marker': 'x' if args.markers else None
        })

if args.max_steps:
    max_x = min(args.max_steps, max_x)
if args.min_steps:
    min_x = max(args.min_steps, min_x)

if args.baseline:
    plots.insert(0, {
        'x': [min_x, max_x],
        'y': [args.baseline] * 2,
        'label': args.baseline_name,
        'color': 'black',
        'linestyle': 'dashed'
    })


lower_is_better = 'loss' in args.metric or 'ppl' in args.metric

if args.sort:
    plots.sort(key=lambda d: (min if lower_is_better else max)(d['y']),
               reverse=not lower_is_better)

for i, plot in enumerate(plots):
    if 'linestyle' not in plot:
        plot['linestyle'] = linestyles[i % len(linestyles)]
    plt.plot(plot.pop('x'), plot.pop('y'), **plot)

if args.xticks:
    plt.xticks(args.xticks)

plt.ylim(ymin=args.min_y, ymax=args.max_y)
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
if args.legend_loc != 'none':
    plt.legend(loc=args.legend_loc)

if args.title:
    plt.title(args.title)

if args.margin:
    left, bottom, right, top = args.margin
    plt.subplots_adjust(left=left, bottom=bottom, right=(1 - right), top=(1 - top), wspace=0, hspace=0)

if args.output:
    plt.savefig(args.output)
    plt.clf()
else:
    plt.show()
