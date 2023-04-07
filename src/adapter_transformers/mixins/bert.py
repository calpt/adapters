import logging
from typing import Callable, Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..lora import Linear as LoRALinear
from ..model_mixin import (
    EmbeddingAdaptersMixin,
    EmbeddingAdaptersWrapperMixin,
    InvertibleAdaptersMixin,
    ModelAdaptersMixin,
    ModelWithHeadsAdaptersMixin,
)
from ..prefix_tuning import PrefixTuningShim


logger = logging.getLogger(__name__)


class BertSelfAttentionAdaptersMixin:
    """Adds adapters to the BertSelfAttention module."""

    def init_adapters(self, config):
        # Wrap layers for LoRA
        self.query = LoRALinear.wrap(self.query, "selfattn", config, attn_key="q")
        self.key = LoRALinear.wrap(self.key, "selfattn", config, attn_key="k")
        self.value = LoRALinear.wrap(self.value, "selfattn", config, attn_key="v")

        self.prefix_tuning = PrefixTuningShim(self.location_key + "_prefix" if self.location_key else None, config)


# For backwards compatibility, BertSelfOutput inherits directly from AdapterLayer
class BertSelfOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the BertSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter")

    def init_adapters(self, config):
        self.location_key = "mh_adapter"
        super().init_adapters(config)


# For backwards compatibility, BertOutput inherits directly from AdapterLayer
class BertOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the BertOutput module."""

    def __init__(self):
        super().__init__("output_adapter")

    def init_adapters(self, config):
        self.location_key = "output_adapter"
        super().init_adapters(config)


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module."""

    def init_adapters(self, config):
        # Wrap layers for LoRA
        self.intermediate.dense = LoRALinear.wrap(self.intermediate.dense, "intermediate", config)
        self.output.dense = LoRALinear.wrap(self.output.dense, "output", config)

        # Set location keys for prefix tuning
        self.attention.self.location_key = "self"
        if self.add_cross_attention:
            self.crossattention.self.location_key = "cross"


class BertModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer

    def hook_after_embeddings(self, hook_fn: Callable):
        return self.embeddings.register_forward_hook(hook_fn)


class BertModelWithHeadsAdaptersMixin(EmbeddingAdaptersWrapperMixin, ModelWithHeadsAdaptersMixin):
    pass
