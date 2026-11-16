"""Align QuaRot draft weights on the host so DSpark can start on Atlas A2.

Problem
-------
vllm-ascend's DSpark draft loader rebuilds the target embedding / LM head and
left-multiplies them by the QuaRot global rotation
(``vllm_ascend.models.qwen3_dspark.align_draft_weights`` ->
``vllm_ascend.models.llama_eagle3.load_quarot_target_layer``). Upstream moves
both the rotation matrix and the shard to the device and runs a float32 MatMul
there. On Atlas A2 that host->device copy of the 2D float32 matrix is lowered to
a ``TransData`` with ``groups=1``, which the AICPU op store rejects:

    EZ3002 Unsupported_Operator: Optype [TransData] of Ops kernel
    [DNN_VM_HOST_CPU_OP_STORE] is unsupported.
    Reason: Transdata op, groups should be greater than 1, but now is 1.

Every worker dies while loading the draft, so the engine cannot start whenever
``--speculative-config method=dspark`` is combined with a QuaRot checkpoint
(w8a8 whose quant description carries ``optional.quarot.rotation_map``).

Fix
---
Compute the alignment entirely in host memory and copy only the finished block
into the device parameter, already cast to the parameter dtype, so the H2D copy
is a plain same-dtype memcpy. Semantics are unchanged: the loaded rows are still
``target_shard.float() @ rotation.float().T`` and the unused tail is zeroed.

This module only registers import hooks, so it is inert until
``vllm_ascend.models.llama_eagle3`` / ``...qwen3_dspark`` are imported, i.e. it
never forces vllm_ascend to be imported early.
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

# ``qwen3_dspark`` does ``from ...llama_eagle3 import load_quarot_target_layer``,
# so both module names need the replacement, not just the defining module.
_SOURCE_MODULE = "vllm_ascend.models.llama_eagle3"
_DSPARK_MODULE = "vllm_ascend.models.qwen3_dspark"

_wrapper = None


def _build_wrapper(module):
    """Build the host-side replacement from the defining module."""
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


def _sweep_aliases():
    """Replace already-bound aliases of the original function."""
    if _wrapper is None:
        return
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("vllm_ascend"):
            continue
        current = getattr(module, "load_quarot_target_layer", None)
        if current is not None and not getattr(
            current, "_ucm_quarot_host_align", False
        ):
            setattr(module, "load_quarot_target_layer", _wrapper)


def _install(module):
    global _wrapper
    if getattr(module, "load_quarot_target_layer", None) is _wrapper:
        return
    if _wrapper is None:
        _wrapper = _build_wrapper(module)
    if _wrapper is None:
        return
    module.load_quarot_target_layer = _wrapper
    _sweep_aliases()


@when_imported(_SOURCE_MODULE)
def _patch_source_module(module):
    try:
        _install(module)
        if _wrapper is not None:
            logger.info(
                "UCM QuaRot host-alignment patch applied to %s", module.__name__
            )
    except Exception as e:  # never break the model import
        logger.error("UCM QuaRot host-alignment patch failed on %s: %s", module.__name__, e)


@when_imported(_DSPARK_MODULE)
def _patch_dspark_module(module):
    try:
        if _wrapper is None:
            source = sys.modules.get(_SOURCE_MODULE)
            if source is not None:
                _install(source)
        if _wrapper is None:
            return
        if getattr(module, "load_quarot_target_layer", None) is not _wrapper:
            module.load_quarot_target_layer = _wrapper
            logger.info(
                "UCM QuaRot host-alignment patch applied to %s", module.__name__
            )
    except Exception as e:  # never break the model import
        logger.error("UCM QuaRot host-alignment patch failed on %s: %s", module.__name__, e)
