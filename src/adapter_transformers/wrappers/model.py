import importlib

from transformers import PreTrainedModel
from transformers.models.auto.auto_factory import getattribute_from_module


def wrap_model(model: PreTrainedModel) -> PreTrainedModel:
    model_type = model.config.model_type
    model_class = model.__class__.__name__
    modules = importlib.import_module(f".{model_type}", "adapter_transformers.models")
    model_class = getattribute_from_module(modules, model_class)
    model.__class__ = model_class
    model._init_adapter_modules()

    return model
