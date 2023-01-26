import os
from typing import Iterable, Tuple, Union, Optional, List
from abc import abstractmethod

import torch.nn as nn

from .composition import AdapterCompositionBlock, Fuse
from .hub_mixin import PushAdapterToHubMixin
from .interfaces import AdaptersInterface
from .loading import WeightsLoader, AdapterLoader, AdapterFusionLoader
from .utils import inherit_doc


@inherit_doc
class MetaModelAdaptersMixin(PushAdapterToHubMixin, AdaptersInterface):
    """Mixin for transformer models adding support for loading/ saving adapters."""
    meta_model_child_names : List[str] = []

    def iter_child_models(self) -> Iterable[AdaptersInterface]:
        return [getattr(self, child_name) for child_name in self.meta_model_child_names]

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        """
        Iterates over all layers of the model.

        This abstract method has to ne implemented by every implementing model.
        """
        for child in self.iter_child_models():
            yield from child.iter_layers()

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock], train_embeddings=False):
        """Sets the model into mode for training the given adapters."""
        for child in self.iter_child_models():
            child.train_adapter(adapter_setup, train_embeddings)

    def train_adapter_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        for child in self.iter_child_models():
            child.train_adapter_fusion(adapter_setup, unfreeze_adapters)

    def has_adapters(self):
        if not getattr(self.config, "is_adaptable", None):
            return False
        return len(self.config.adapters.adapters) > 0

    @property
    def has_parallel_adapters(self) -> bool:
        if self.config.adapters.active_setup:
            return self.config.adapters.active_setup.parallel_channels > 1
        else:
            return False

    @property
    def active_adapters(self) -> AdapterCompositionBlock:
        return self.config.adapters.active_setup

    def set_active_adapters(
        self, adapter_setup: Union[list, AdapterCompositionBlock], skip_layers: Optional[List[int]] = None
    ):
        """
        Sets the adapter modules to be used by default in every forward pass. If no adapter with the given name is
        found, no module of the respective type will be activated.

        Args:
            adapter_setup (list):
                The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        # TODO-AH: Apply skip_layers logic
        for child in self.iter_child_models():
            child.set_active_adapters(adapter_setup, skip_layers)

    def add_adapter(self, adapter_name: str, config=None, overwrite_ok: bool = False, set_active: bool = False):
        """
        Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            config (str or dict or AdapterConfigBase, optional): The adapter configuration, can be either:

                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
                - a list of configs where each item matches the config to be used in one of this model's child models.
            overwrite_ok (bool, optional):
                Overwrite an adapter with the same name if it exists. By default (False), an exception is thrown.
            set_active (bool, optional):
                Set the adapter to be the active one. By default (False), the adapter is added but not activated.
        """
        if not isinstance(config, list):
            config = [config] * len(self.meta_child_model_names)
        for child, child_config in zip(self.iter_child_models(), config):
            child.add_adapter(adapter_name, child_config, overwrite_ok, set_active)

    def add_adapter_fusion(
        self,
        adapter_names: Union[Fuse, list, str],
        config=None,
        overwrite_ok: bool = False,
        set_active: bool = False,
    ):
        """
        Adds AdapterFusion to the model with alll the necessary configurations and weight initializations

        Args:
            adapter_names (Fuse or list or str): AdapterFusion layer to add. Can be either:

                - a ``Fuse`` composition block
                - a list of adapter names to fuse
                - a comma-separated string of adapter names to fuse
            config (str or dict): adapter fusion configuration, can be either:

                - a string identifying a pre-defined adapter fusion configuration
                - a dictionary representing the adapter fusion configuration
                - the path to a file containing the adapter fusion configuration
                - a list of configs where each item matches the config to be used in one of this model's child models.
            overwrite_ok (bool, optional):
                Overwrite an AdapterFusion layer with the same name if it exists. By default (False), an exception is
                thrown.
            set_active (bool, optional):
                Activate the added AdapterFusion. By default (False), the AdapterFusion is added but not activated.
        """
        if not isinstance(config, list):
            config = [config] * len(self.meta_child_model_names)
        for child, child_config in zip(self.iter_child_models(), config):
            child.add_adapter_fusion(adapter_names, child_config, overwrite_ok, set_active)

    def delete_adapter(self, adapter_name: str):
        for child in self.iter_child_models():
            child.delete_adapter(adapter_name)

    def delete_adapter_fusion(self, adapter_names: Union[Fuse, list, str]):
        for child in self.iter_child_models():
            child.delete_adapter_fusion(adapter_names)

    def save_adapter(
        self,
        save_directory: str,
        adapter_name: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        os.makedirs(save_directory, exist_ok=True)
        for child_name in self.meta_child_model_names:
            child = getattr(self, child_name)
            save_directory = os.path.join(save_directory, child_name)
            child.save_adapter(save_directory, adapter_name, meta_dict, custom_weights_loaders)

    def save_adapter_fusion(
        self,
        save_directory: str,
        adapter_names: Union[Fuse, list, str],
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        os.makedirs(save_directory, exist_ok=True)
        for child_name in self.meta_child_model_names:
            child = getattr(self, child_name)
            save_directory = os.path.join(save_directory, child_name)
            child.save_adapter_fusion(save_directory, adapter_names, meta_dict, custom_weights_loaders)

    def load_adapter(
        self,
        adapter_name_or_path: str,
        config: Union[dict, str] = None,
        version: str = None,
        model_name: str = None,
        load_as: str = None,
        source: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        leave_out: Optional[List[int]] = None,
        id2label=None,
        set_active: bool = False,
        **kwargs
    ) -> str:
        for child_name in self.meta_child_model_names:
            child = getattr(self, child_name)
            
            child.load_adapter(
                adapter_name_or_path,
                config,
                version,
                model_name,
                load_as,
                source,
                custom_weights_loaders,
                leave_out,
                id2label,
                set_active,
                **kwargs
            )

    def load_adapter_fusion(
        self,
        adapter_fusion_name_or_path: str,
        load_as: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        set_active: bool = False,
        **kwargs
    ) -> str:
        """
        Loads a pre-trained AdapterFusion layer from the local file system.

        Args:
            adapter_fusion_name_or_path (str):
                a path to a directory containing AdapterFusion weights saved using `model.save_adapter_fusion()`.
            load_as (str, optional): Load the AdapterFusion using this name.
                    By default, the name with which the AdapterFusion layer was saved will be used.
            set_active (bool, optional):
                Activate the loaded AdapterFusion. By default (False), the AdapterFusion is loaded but not activated.

        Returns:
            str: The name with which the AdapterFusion was added to the model.
        """
        loader = AdapterFusionLoader(self)
        load_dir, load_name = loader.load(adapter_fusion_name_or_path, load_as, set_active=set_active)
        # load additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(
                    load_dir,
                    load_as=load_as,
                    loading_info=kwargs.get("loading_info", None),
                    main_load_name=load_name,
                    set_active=set_active,
                )
        return load_name

    def save_all_adapters(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves all adapters of this model together with their configuration to subfolders of the given location.

        Args:
            save_directory (str): Path to a directory where the adapters should be saved.
        """
        pass

    @abstractmethod
    def save_all_adapter_fusions(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves all AdapterFusion layers of this model together with their configuration to subfolders of the given
        location.

        Args:
            save_directory (str): Path to a directory where the AdapterFusion layers should be saved.
        """
        pass

    @abstractmethod
    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        pass

    @abstractmethod
    def get_adapter(self, name) -> dict:
        """
        Returns a dictionary with all weights of the adapter with the specified name.

        Args:
            name (str): The adapter name.

        Returns:
            dict: A nested dictionary containing the weights of the adapter. The dictionary is structured as follow:
            {<layer id>: {<module location>: <nn.Module>}}. <layer id> = -1 indicates global/ shared weights.
        """
        pass

    @abstractmethod
    def adapter_summary(self, as_dict=False) -> Union[str, dict]:
        """
        Returns a string summary of all adapters currently added to the model. Each entry in the summary table has the
        following attributes:

            - name: the name of the adapter
            - architecture: the architectural base of the adapter
            - #param: the number of parameters of the adapter
            - %param: the number of parameters of the adapter relative to the full model
            - active: whether the adapter is active
            - train: whether the adapter weights are enabled for training
        """
        pass

    @abstractmethod
    def eject_prefix_tuning(self, name: str):
        """
        Converts the prefix tuning with the given name from the reparameterized form into the flat form.

        Args:
            name (str): The name of the prefix tuning.
        """
        pass

    @abstractmethod
    def merge_adapter(self, name: str):
        """
        Merges the weights of the given LoRA module with the Transformer weights as described in the paper.

        Args:
            name (str): LoRA module to merge.
        """
        pass

    @abstractmethod
    def reset_adapter(self):
        """
        Resets weights of a LoRA module merged using `model.merge_adapter(name)`.
        """
        pass
