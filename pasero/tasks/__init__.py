# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

from .task import Corpus, InferenceCorpus, Task
from .translation import ParallelCorpus, InferenceParallelCorpus, TranslationTask
from .language_modeling import MonolingualCorpus, LanguageModelingTask
from .speech_translation import SpeechTranslationTask
from .doc_level_translation import DocumentLevelTranslationTask
from .nllb_translation import NLLBTranslationTask
from .dialogue import DialogueTask


def get_task_class(task: str) -> type[Task]:
    if task == 'translation':
        return TranslationTask
    elif task == 'speech_translation':
        return SpeechTranslationTask
    elif task == 'language_modeling':
        return LanguageModelingTask
    elif task == 'doc_level_translation':
        return DocumentLevelTranslationTask
    elif task == 'nllb_translation':
        return NLLBTranslationTask
    elif task == 'dialogue':
        return DialogueTask
    else:
        raise NotImplementedError
