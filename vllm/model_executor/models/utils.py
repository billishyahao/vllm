import itertools
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Literal, Mapping, Optional,
                    Protocol, Tuple, Union, overload)

import torch
import torch.nn as nn
from torch.func import functional_call
from transformers import PretrainedConfig

from vllm.config import (CacheConfig, LoRAConfig, MultiModalConfig,
                         SchedulerConfig)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.loader import build_model
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import ModelRegistry
from vllm.multimodal.base import NestedTensors
from vllm.sequence import IntermediateTensors
from vllm.utils import is_pin_memory_available

WeightsMapping = Mapping[str, Optional[str]]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def apply(self, key: str) -> Optional[str]:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key


class AutoWeightsLoader:
    """
    Helper class to load weights into a :class:`torch.nn.Module`.

    The weight loading logic for individual modules can be overridden
    by defining a ``load_weights`` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a ``weight_loader`` method.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        allow_missing_prefixes: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.allow_missing_prefixes = allow_missing_prefixes or []

    def _is_strict(self, prefix: str, rest: str) -> bool:
        weight_name = prefix if rest == "" else ".".join((prefix, rest))

        return not any(
            weight_name.startswith(p) for p in self.allow_missing_prefixes)

    def _load_param(
        self,
        prefix: str,
        param: nn.Parameter,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> None:
        for weight_name, weight_data in weights:
            if weight_name != "":
                if self._is_strict(prefix, weight_name):
                    raise ValueError(
                        f"Attempted to load nested weight ({weight_name}) "
                        f"into a single parameter ({prefix})")

                continue

            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, weight_data)

    def _load_module(
        self,
        prefix: str,
        module: nn.Module,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> None:
        # TODO: Recursive search for load_weights
        module_load_weights = getattr(module, "load_weights", None)
        if callable(module_load_weights):
            module_load_weights(weights)

        params_dict = dict(module.named_parameters())

        for weight_name, weight_data in weights:
            if is_pp_missing_parameter(weight_name, module):
                continue

            if weight_name not in params_dict:
                if self._is_strict(prefix, weight_name):
                    raise ValueError(
                        f"Attempted to load unexpected weight ({weight_name}) "
                        f"into module ({prefix}). "
                        f"Available weights: {list(params_dict.keys())}")

                continue

            param = params_dict[weight_name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, weight_data)

    def _groupby_prefix(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, Iterable[Tuple[str, torch.Tensor]]]]:
        weights_by_parts = ((weight_name.split(".", 1), weight_data)
                            for weight_name, weight_data in weights)

        for prefix, group in itertools.groupby(weights_by_parts,
                                               key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1, the length of `rest` can either be 0 or 1
                (("" if len(rest) == 0 else rest[0], weights_data)
                 for (_, *rest), weights_data in group),
            )

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        *,
        mapper: Optional[WeightsMapper] = None,
    ) -> None:
        child_modules = dict(self.module.named_children())
        child_params = dict(self.module.named_parameters(recurse=False))

        if mapper is not None:
            weights = ((out_name, data) for name, data in weights
                       if (out_name := mapper.apply(name)) is not None)

        for prefix, child_weights in self._groupby_prefix(weights):
            if prefix in child_modules:
                self._load_module(prefix, child_modules[prefix], child_weights)
            elif prefix in child_params:
                self._load_param(prefix, child_params[prefix], child_weights)
            else:
                msg = f"There is no child module or parameter named {prefix}"
                raise ValueError(msg)


def init_vllm_registered_model(
    hf_config: PretrainedConfig,
    cache_config: Optional[CacheConfig],
    quant_config: Optional[QuantizationConfig],
    *,
    lora_config: Optional[LoRAConfig] = None,
    multimodal_config: Optional[MultiModalConfig] = None,
    scheduler_config: Optional[SchedulerConfig] = None,
) -> nn.Module:
    """
    Helper function to initialize an inner model registered to vLLM,
    based on the arguments passed to the outer vLLM model.
    """
    model_class, _ = ModelRegistry.resolve_model_cls(hf_config.architectures)

    return build_model(
        model_class,
        hf_config,
        cache_config,
        quant_config,
        lora_config=lora_config,
        multimodal_config=multimodal_config,
        scheduler_config=scheduler_config,
    )


@overload
def flatten_bn(x: torch.Tensor) -> torch.Tensor:
    ...


@overload
def flatten_bn(x: List[torch.Tensor]) -> List[torch.Tensor]:
    ...


@overload
def flatten_bn(
    x: Union[List[torch.Tensor], torch.Tensor],
    *,
    concat: Literal[True],
) -> torch.Tensor:
    ...


def flatten_bn(
    x: Union[List[torch.Tensor], torch.Tensor],
    *,
    concat: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Flatten the ``B`` and ``N`` dimensions of batched multimodal inputs.

    The input tensor should have shape ``(B, N, ...)```.
    """
    if isinstance(x, torch.Tensor):
        return x.flatten(0, 1)

    if concat:
        return torch.cat(x)

    return [x_n for x_b in x for x_n in x_b]


def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(
        _embedding_count_expression(inner) for inner in embeddings)


def merge_multimodal_embeddings(input_ids: torch.Tensor,
                                inputs_embeds: torch.Tensor,
                                multimodal_embeddings: NestedTensors,
                                placeholder_token_id: int) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    mask = (input_ids == placeholder_token_id)
    num_expected_tokens = mask.sum().item()
    assert isinstance(num_expected_tokens, int)

    flattened = _flatten_embeddings(multimodal_embeddings)
    if flattened.shape[0] != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {flattened.shape[0]} "
            f"multimodal tokens to {num_expected_tokens} placeholders")

    inputs_embeds[mask] = flattened
    return inputs_embeds


class LayerFn(Protocol):

    def __call__(self, prefix: str) -> torch.nn.Module:
        ...


class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()


_CPU_OFFLOAD_BYTES = 0
_CPU_OFFLOAD_MAX_BYTES = 0


def set_cpu_offload_max_bytes(max_bytes: int) -> None:
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    _CPU_OFFLOAD_BYTES = 0
    _CPU_OFFLOAD_MAX_BYTES = max_bytes


def maybe_offload_to_cpu(module: torch.nn.Module) -> torch.nn.Module:
    device = next(module.parameters()).device

    if device == torch.device("cpu"):
        return module

    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        return module

    pin_memory = is_pin_memory_available()

    # offload parameters to CPU
    # use pin_memory if possible, which helps cudagraph capture speed
    offloaded_parameters = False
    for p in module.parameters():
        if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
            # we use per-parameter offloading
            # one module might have some parameters offloaded and some not
            break

        # `torch.empty_like` does not support `pin_memory` argument
        cpu_data = torch.empty_strided(size=p.data.size(),
                                       stride=p.data.stride(),
                                       dtype=p.data.dtype,
                                       layout=p.data.layout,
                                       device='cpu',
                                       pin_memory=pin_memory)
        cpu_data.copy_(p.data)
        p.data = cpu_data
        _CPU_OFFLOAD_BYTES += p.data.numel() * p.data.element_size()
        offloaded_parameters = True

    if offloaded_parameters:
        original_forward = module.forward

        def forward(*args, **kwargs):
            module.forward = original_forward
            device_state = {
                # here we blindly call `to(device)`
                # if the parameter is already on the device, it will be a no-op
                k: v.to(device, non_blocking=True)
                for k, v in module.state_dict().items()
            }
            output = functional_call(module,
                                     device_state,
                                     args=args,
                                     kwargs=kwargs)
            module.forward = forward
            return output

        module.forward = forward

    return module


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
) -> Tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices
    start_layer, end_layer = get_pp_indices(num_hidden_layers,
                                            get_pp_group().rank_in_group,
                                            get_pp_group().world_size)
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)] + [
            maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
            for idx in range(start_layer, end_layer)
        ] + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)])
    return start_layer, end_layer, modules


# NOTE: don't use lru_cache here because it can prevent garbage collection
_model_to_pp_missing_layer_names: Dict[int, List[str]] = {}


def get_pp_missing_layer_names(model: torch.nn.Module) -> List[str]:
    """Get the names of the missing layers in a pipeline parallel model."""
    model_id = id(model)
    if model_id in _model_to_pp_missing_layer_names:
        return _model_to_pp_missing_layer_names[model_id]

    missing_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, PPMissingLayer):
            # NOTE: the trailing dot is used to match the prefix of the layer.
            # without the dot, we could match a layer that is not missing,
            # e.g., 'encoder.layer.1' would match 'encoder.layer.11'
            missing_layer_names.append(name + '.')
    _model_to_pp_missing_layer_names[model_id] = missing_layer_names

    return missing_layer_names


def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool:
    """Check if a parameter is missing in a pipeline parallel model."""
    if isinstance(model, PPMissingLayer):
        return True

    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in get_pp_missing_layer_names(model))


def make_empty_intermediate_tensors_factory(keys: List[str], hidden_size: int):

    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors({
            key: torch.zeros((batch_size, hidden_size),
                             dtype=dtype,
                             device=device)
            for key in keys
        })

    return make_empty_intermediate_tensors


class LLMWrapper(nn.Module):
    """
    To align with the key names of LoRA trained with PEFT, we need to add an 
    additional layer to the llm's implementation.
    """

    def __init__(self, llm: nn.Module, name: str) -> None:
        super().__init__()
        self.model_name = name
        setattr(self, name, llm)

    def __getattr__(self, key: str):
        llm = super().__getattr__(self.model_name)
        if key == self.model_name:
            return llm

        return getattr(llm, key)

    # We need to explicitly override this
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        llm = super().__getattr__(self.model_name)
        return llm(*args, **kwargs)
