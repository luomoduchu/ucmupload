"""Align QuaRot draft weights on the host so DSpark can start on Atlas A2.

Problem
-------
vllm-ascend rotates the draft's embedding / LM head / projection weights on the
device when the target checkpoint is QuaRot-quantized (w8a8 whose quant
description carries ``optional.quarot.rotation_map``). Three call sites move a
2-D float32 tensor with ``Tensor.to(device="npu")``:

* ``llama_eagle3.load_quarot_target_layer`` — ``rotation.to(device=...)``;
* ``qwen3_dspark.process_weight`` — ``rotation_weight.to(device=...)`` when the
  caller passed a device tensor;
* ``qwen3_dspark.align_draft_weights`` —
  ``weight.copy_(process_weight(weight.cpu(), rotation).to(weight.device))``.

On Atlas A2 that host->device copy of a 2-D matrix is lowered by GE to a
``TransData`` node (op name ``Identity``) with ``groups=1``, which the AICPU op
store rejects, so the worker dies while loading the draft:

    EZ3002 Unsupported_Operator: Optype [TransData] of Ops kernel
    [DNN_VM_HOST_CPU_OP_STORE] is unsupported.
    Reason: Transdata op, groups should be greater than 1, but now is 1.

Plain ``param.copy_(cpu_tensor)`` (what vLLM's weight loader does for every
model on every start) is *not* affected, and neither is ``tensor.cpu()``.

Fix
---
Keep every rotation in host memory and write the finished, already
dtype-cast block back with ``copy_``, so the only remaining host->device hop is
a same-dtype memcpy:

* ``load_quarot_target_layer`` is replaced wholesale (chunked host alignment,
  4096 rows at a time, unused tail zeroed);
* ``align_draft_weights`` is replaced with a host-safe copy of upstream's logic
  (same ownership rules, same PP fallback) that rotates the projection with
  ``process_weight(weight.cpu(), rotation)`` and writes it back with
  ``weight.data.copy_(...)`` instead of ``.to(weight.device)``.

Semantics are unchanged: ``target_shard.float() @ rotation.float().T`` truncated
to the shard rows, tail rows zeroed. The fp32 intermediates stay in host memory,
so the full-vocabulary product no longer has to fit in device memory either.

This module only registers import hooks, so it is inert until
``vllm_ascend``'s DSpark / DFlash modules are imported.
"""

from __future__ import annotations

import sys
from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

# Rows per host-side MatMul/copy step; bounds the fp32 temporary for a full
# vocabulary shard without making the alignment slow.
_ALIGN_CHUNK_ROWS = 4096

# ``qwen3_dspark`` does ``from ...llama_eagle3 import load_quarot_target_layer``
# and the DFlash/Kimi/GLM drafts do ``from ...qwen3_dspark import
# align_draft_weights``, so every one of these module attributes needs the
# replacement, not just the defining module. ``quarot_utils`` is the module the
# upstream refactor (vllm-ascend#21783) is moving those helpers into.
_PATCHED_MODULES = (
    "vllm_ascend.models.llama_eagle3",
    "vllm_ascend.models.qwen3_dspark",
    "vllm_ascend.models.qwen3_dflash2",
    "vllm_ascend.models.kimi_k3_dspark",
    "vllm_ascend.models.glm5_dspark",
    "vllm_ascend.models.quarot_utils",
)

_SOURCE_MODULE = "vllm_ascend.models.llama_eagle3"
_DSPARK_MODULE = "vllm_ascend.models.qwen3_dspark"

_loader_wrapper = None
_align_wrapper = None

_TARGET_EMBED_WEIGHT_NAMES = (
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
_TARGET_LM_HEAD_WEIGHT_NAMES = (
    "language_model.lm_head.weight",
    "lm_head.weight",
)


def _build_loader_wrapper(module):
    """Build the host-side replacement for ``load_quarot_target_layer``."""
    import torch
    from safetensors import safe_open

    original = getattr(module, "load_quarot_target_layer", None)
    find_weight = getattr(module, "_find_safetensors_weight", None)
    if not callable(original) or not callable(find_weight):
        logger.warning(
            "UCM QuaRot host alignment skipped: %s exposes no "
            "load_quarot_target_layer/_find_safetensors_weight",
            module.__name__,
        )
        return None

    from pathlib import Path

    @wraps(original)
    @torch.inference_mode()
    def load_quarot_target_layer(
        layer, target_model_path, weight_names, rotation, label
    ):
        target_model_path = Path(target_model_path)
        shard_path, weight_name = find_weight(target_model_path, weight_names)

        shard_indices = getattr(layer, "shard_indices", None)
        if shard_indices is None:
            start_index = 0
            end_index = layer.weight.shape[0]
        else:
            start_index = shard_indices.org_vocab_start_index
            end_index = shard_indices.org_vocab_end_index

        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            target_weight = shard.get_slice(weight_name)[
                start_index:end_index
            ].to(dtype=torch.float32)

        rotation_t = None
        if rotation is not None:
            rotation_t = rotation.cpu().to(dtype=torch.float32).T.contiguous()

        loaded_rows = target_weight.shape[0]
        dtype = layer.weight.dtype
        for start in range(0, loaded_rows, _ALIGN_CHUNK_ROWS):
            stop = min(start + _ALIGN_CHUNK_ROWS, loaded_rows)
            block = target_weight[start:stop]
            if rotation_t is not None:
                block = torch.matmul(block, rotation_t)
            layer.weight.data[start:stop].copy_(block.to(dtype=dtype))
        layer.weight.data[loaded_rows:].zero_()

        logger.info(
            "UCM QuaRot host alignment: loaded %s from %s %s (rotated=%s)",
            label,
            shard_path.name,
            tuple(layer.weight.shape),
            rotation_t is not None,
        )

    load_quarot_target_layer._ucm_quarot_host_align = True
    return load_quarot_target_layer


def _build_align_wrapper(module):
    """Build the host-side replacement for ``align_draft_weights``.

    ``module`` must expose ``process_weight``; the body mirrors upstream's
    ownership rules so that only the projection write-back changes.
    """
    import torch
    from vllm.distributed import get_pp_group
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm_ascend.utils import get_rotation_matrix, get_rotation_path

    original = getattr(module, "align_draft_weights", None)
    process_weight = getattr(module, "process_weight", None)
    if not callable(original) or not callable(process_weight):
        logger.warning(
            "UCM QuaRot host alignment skipped: %s exposes no "
            "align_draft_weights/process_weight",
            module.__name__,
        )
        return None

    embed_weight_names = getattr(
        module, "TARGET_EMBED_WEIGHT_NAMES", _TARGET_EMBED_WEIGHT_NAMES
    )
    lm_head_weight_names = getattr(
        module, "TARGET_LM_HEAD_WEIGHT_NAMES", _TARGET_LM_HEAD_WEIGHT_NAMES
    )

    def _rotate_projection(weight, rotation):
        # ``.to(weight.device)`` on the fp32 product is the op that fails on A2;
        # rotate in host memory and write back with a same-dtype copy instead.
        aligned = process_weight(weight.cpu(), rotation)
        weight.data.copy_(aligned)
        logger.info(
            "UCM QuaRot host alignment: rotated draft projection %s",
            tuple(weight.shape),
        )

    @wraps(original)
    @torch.no_grad()
    def align_draft_weights(model, projection, vllm_config):
        """Align draft inputs with the rotated target without modifying shared weights."""
        loader = sys.modules[_SOURCE_MODULE].load_quarot_target_layer
        rotation_path = get_rotation_path(vllm_config)
        if rotation_path is not None:
            rotation = get_rotation_matrix(rotation_path).cpu()
            _rotate_projection(projection.weight, rotation)
        else:
            if get_pp_group().world_size == 1:
                return
            # PP stages past the first cannot alias the target's stage-0
            # embedding; load this rank's vocab shard from the target
            # checkpoint unrotated so the draft owns its copy.
            rotation = None

        weight = projection.weight
        target_config = vllm_config.model_config.hf_text_config
        for owner, name, layer_cls, weight_names, own_flag in (
            (model.model, "embed_tokens", VocabParallelEmbedding,
             embed_weight_names, "has_own_embed_tokens"),
            (model, "lm_head", ParallelLMHead,
             lm_head_weight_names, "has_own_lm_head"),
        ):
            if getattr(model, own_flag, False):
                continue
            if rotation is None and name != "embed_tokens":
                # The target lm_head lives on the last PP stage and can still
                # be aliased; only the embedding needs its own copy.
                continue
            with torch.device(weight.device):
                layer = layer_cls(
                    target_config.vocab_size,
                    target_config.hidden_size,
                    params_dtype=weight.dtype,
                )
            loader(
                layer,
                vllm_config.model_config.model,
                weight_names,
                rotation,
                f"draft {name}.weight",
            )
            layer.quant_method.process_weights_after_loading(layer)
            setattr(owner, name, layer)
            setattr(model, own_flag, True)

    align_draft_weights._ucm_quarot_host_align = True
    return align_draft_weights


def _sweep_aliases():
    """Replace already-bound aliases of the upstream functions."""
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("vllm_ascend"):
            continue
        _install(module, quiet=True)


def _install(module, quiet=False):
    global _loader_wrapper, _align_wrapper
    changed = []

    loader = getattr(module, "load_quarot_target_layer", None)
    if loader is not None and not getattr(
        loader, "_ucm_quarot_host_align", False
    ):
        if _loader_wrapper is None:
            _loader_wrapper = _build_loader_wrapper(
                sys.modules.get(_SOURCE_MODULE, module)
            )
        if _loader_wrapper is not None:
            module.load_quarot_target_layer = _loader_wrapper
            changed.append("load_quarot_target_layer")

    align = getattr(module, "align_draft_weights", None)
    if align is not None and not getattr(align, "_ucm_quarot_host_align", False):
        if _align_wrapper is None:
            _align_wrapper = _build_align_wrapper(
                sys.modules.get(_DSPARK_MODULE, module)
            )
        if _align_wrapper is not None:
            module.align_draft_weights = _align_wrapper
            changed.append("align_draft_weights")

    if changed and not quiet:
        logger.info(
            "UCM QuaRot host-alignment patch applied to %s (%s)",
            module.__name__,
            ", ".join(changed),
        )
    return bool(changed)


def _make_hook():
    def _hook(module):
        try:
            _install(module)
            _sweep_aliases()
        except Exception as e:  # never break the model import
            logger.error(
                "UCM QuaRot host-alignment patch failed on %s: %s",
                module.__name__,
                e,
            )

    return _hook


_HOOK = _make_hook()

for _module_name in _PATCHED_MODULES:
    when_imported(_module_name)(_HOOK)
