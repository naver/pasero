# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import logging
from pasero.config import *
from .transformer import Transformer, Encoder, Decoder, EncoderDecoder
from .adapters import AdapterTransformer
from .hybrid import HybridTransformer, AdapterHybridTransformer
from .mixture_of_experts import MOETransformer, get_moe_cls
from .modules import fast_init

logger = logging.getLogger('models')

ARCHITECTURES = {
    'transformer': Transformer,
    'adapter_transformer': AdapterTransformer,
    'hybrid_transformer': HybridTransformer,
    'adapter_hybrid_transformer': AdapterHybridTransformer,
    'moe_transformer': MOETransformer,
}

def get_architecture(model_cfg: TransformerConfig) -> type[Transformer]:
    # maps architecture names (as specified by --arch) to Modules
    base_arch = model_cfg._arch  # e.g., TransformerSmallConfig -> transformer
    return ARCHITECTURES[base_arch]
