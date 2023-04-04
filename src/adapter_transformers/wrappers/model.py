import importlib

from transformers import PreTrainedModel
from transformers.models.auto.auto_factory import getattribute_from_module
from transformers.models.auto.configuration_auto import model_type_to_module_name

from ..model_mixin import ModelAdaptersMixin


def wrap_model(model: PreTrainedModel) -> PreTrainedModel:
    if isinstance(model, ModelAdaptersMixin):
        return model
    model_name = model_type_to_module_name(model.config.model_type)
    modules = importlib.import_module(f".{model_name}.modeling_{model_name}", "adapter_transformers.models")
    for module in model.modules():
        if module.__class__.__module__.startswith("transformers.models"):
            module_class = getattribute_from_module(modules, module.__class__.__name__)
            module.__class__ = module_class
    model.init_adapters(model.config)

    return model
