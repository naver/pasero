from setuptools import setup, find_packages

setup(
    name='pasero',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.1',
        'sacrebleu>=2.3.0',
        'regex',
        'numpy',
        'mlflow',
        'sentencepiece',
        'psutil',
        'flask',
        'waitress',
        'pyyaml',
        'fasttext',
        'matplotlib',
        'tqdm',
        'emoji',
        'transformers',
        'python-dateutil',
        'jiwer',
        'stopes',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'pasero-train = cli.train:main',
            'pasero-decode = cli.decode:main',
            'pasero-serve = cli.serve:main',
            'pasero-serve-hf = cli.serve_hf:main',
            # Pasero-Tokenizer actions
            'pasero-tokenize = cli.tokenizer:main_tokenize',
            'pasero-detokenize = cli.tokenizer:main_detokenize',
            'pasero-build-dict = cli.tokenizer:main_build_dict',
            'pasero-build-tokenizer = cli.tokenizer:main_train',
            'pasero-noisify = cli.tokenizer:main_noisify',
        ],
    },
)