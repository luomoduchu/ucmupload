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

Three layers, so that no QuaRot weight path keeps the broken transfer:

1. ``load_quarot_target_layer`` — replaced (chunked host alignment).
2. ``align_draft_weights`` — replaced (host rotate + ``copy_`` write-back),
   rebound on every draft module that imported it by name.
3. ``process_weight`` — replaced so that a *device* input also rotates on the
   host and returns through ``copy_`` instead of a device-side transfer, plus
   ``safe_tensor_to``: a size-gated net on ``Tensor.to`` / ``Tensor.npu`` that
   routes any remaining CPU->NPU move of a large tensor through
   ``empty(...) + copy_`` (that is the op pair the A2 op store does support).
   The net covers sites this module does not know by name (other model families,
   quant post-processing, future upstream code); disable it with
   ``UCM_QUAROT_SAFE_TO=0``.

This module only registers import hooks (and the ``Tensor.to`` net), so it is
inert until ``vllm_ascend``'s model modules are imported.
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

# `safe_tensor_to` only re-routes host->device moves at least this large. The
# broken TransData shows up for weight/rotation matrices (16M elements and up);
# small per-step transfers (ids, masks, quant scales) keep the stock path, and
# so does everything on device already.
_SAFE_TO_MIN_NUMEL = 1 << 20

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
_process_weight_wrapper = None
_safe_to_installed = False
_safe_to_logged = False

_TARGET_EMBED_WEIGHT_NAMES = (
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
_TARGET_LM_HEAD_WEIGHT_NAMES = (
    "language_model.lm_head.weight",
    "lm_head.weight",
)

# Host->device writes are issued in blocks of at most this many bytes. The A2
# op store rejects the single-op model it builds for one huge ND->NZ TransData
# (`EZ3002 ... groups should be greater than 1`), so keep every transfer small
# enough that it is a plain memcpy, and anchor it to a fresh ND tensor.
_COPY_CHUNK_BYTES = 8 << 20
_COPY_SYNC_EVERY_CHUNK = False


def _npu_format(tensor):
    """Return the ACL format of a device tensor, or None when unavailable."""
    try:
        import torch_npu

        return torch_npu.get_npu_format(tensor)
    except Exception:
        return None


def _weight_nz_mode():
    """Return ``additional_config.weight_nz_mode`` (0/1/2) or None."""
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return int(get_ascend_config().weight_nz_mode)
    except Exception:
        return None


def _apply_weight_layout(tensor):
    """Re-apply the layout the layer expects (FRACTAL_NZ when configured).

    Mirrors what ``process_weights_after_loading`` does to a freshly loaded
    weight. ``maybe_trans_nz`` is a no-op for dtypes/layers that stay in ND, so
    this is idempotent and never double-quantizes.
    """
    try:
        from vllm_ascend.utils import maybe_trans_nz
    except Exception:
        return tensor
    try:
        return maybe_trans_nz(tensor)
    except Exception as e:
        logger.warning("UCM QuaRot host alignment: layout cast failed: %s", e)
        return tensor


def _chunk_rows(width, element_size):
    """Rows per host->device block, sized from ``_COPY_CHUNK_BYTES``."""
    row_bytes = max(1, int(width) * max(1, int(element_size)))
    return max(1, _COPY_CHUNK_BYTES // row_bytes)


def _write_host_block(dst, src):
    """Copy a host tensor into a device tensor in small blocks.

    Returns True when ``dst`` is the destination that was written.
    """
    import torch

    if src.shape != dst.shape:
        dst.copy_(src)
        return True
    rows = _chunk_rows(dst.shape[-1], dst.element_size())
    if dst.ndim < 2 or dst.shape[0] <= rows:
        dst.copy_(src)
        return True
    for start in range(0, dst.shape[0], rows):
        stop = min(start + rows, dst.shape[0])
        dst[start:stop].copy_(src[start:stop])
        if _COPY_SYNC_EVERY_CHUNK and torch.npu.is_available():
            torch.npu.synchronize()
    return True


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
        rows = max(
            1,
            min(
                _ALIGN_CHUNK_ROWS,
                _chunk_rows(layer.weight.shape[1], layer.weight.element_size()),
            ),
        )
        for start in range(0, loaded_rows, rows):
            stop = min(start + rows, loaded_rows)
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


def _build_process_weight_wrapper(module):
    """Make ``process_weight`` rotate on the host whatever it is handed.

    Upstream does ``rotation.to(device=linear_weight.device)`` first, so a
    device input transfers the rotation matrix. Keep that contract (return a
    tensor on the caller's device) but do the math in host memory and hand the
    result back through ``empty(...) + copy_``.
    """
    import torch

    original = getattr(module, "process_weight", None)
    if not callable(original):
        logger.warning(
            "UCM QuaRot host alignment skipped: %s exposes no process_weight",
            module.__name__,
        )
        return None

    @wraps(original)
    def process_weight(linear_weight, rotation_weight):
        if linear_weight.device.type == "cpu":
            return original(linear_weight, rotation_weight)
        host = linear_weight.cpu()
        if rotation_weight.device.type != "cpu":
            rotation_weight = rotation_weight.cpu()
        aligned = original(host, rotation_weight)
        out = torch.empty(
            aligned.shape, dtype=aligned.dtype, device=linear_weight.device
        )
        _write_host_block(out, aligned)
        logger.info(
            "UCM QuaRot host alignment: rotated %s weight %s",
            linear_weight.device,
            tuple(linear_weight.shape),
        )
        return out

    process_weight._ucm_quarot_host_align = True
    return process_weight


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

    def _rotate_projection(projection, rotation):
        # Writing into the live parameter directly is what fails on A2: after
        # `process_weights_after_loading` the weight may already be
        # FRACTAL_NZ (weight_nz_mode), and an ND host->NZ device copy lowers to
        # a TransData whose `groups` is 1, which the AICPU op store rejects.
        # So rotate in host memory, land the result in a *fresh ND* buffer in
        # small blocks, then re-apply the layout the layer expects - exactly the
        # sequence a normal load followed by `process_weights_after_loading`
        # performs.
        weight = projection.weight
        aligned = module.process_weight(weight.cpu(), rotation)
        target = weight.data
        format_before = _npu_format(target)
        fresh = torch.empty(target.shape, dtype=target.dtype, device=target.device)
        _write_host_block(fresh, aligned)
        if getattr(projection, "skip_weight_nz_conversion", False):
            pass
        else:
            fresh = _apply_weight_layout(fresh)
        weight.data = fresh
        logger.info(
            "UCM QuaRot host alignment: rotated draft projection %s "
            "(acl format %s -> %s, nz_mode=%s)",
            tuple(weight.shape),
            format_before,
            _npu_format(fresh),
            _weight_nz_mode(),
        )

    @wraps(original)
    @torch.no_grad()
    def align_draft_weights(model, projection, vllm_config):
        """Align draft inputs with the rotated target without modifying shared weights."""
        loader = sys.modules[_SOURCE_MODULE].load_quarot_target_layer
        rotation_path = get_rotation_path(vllm_config)
        if rotation_path is not None:
            rotation = get_rotation_matrix(rotation_path).cpu()
            _rotate_projection(projection, rotation)
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
    global _loader_wrapper, _align_wrapper, _process_weight_wrapper
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

    process_weight = getattr(module, "process_weight", None)
    if process_weight is not None and not getattr(
        process_weight, "_ucm_quarot_host_align", False
    ):
        if _process_weight_wrapper is None:
            _process_weight_wrapper = _build_process_weight_wrapper(module)
        if _process_weight_wrapper is not None:
            module.process_weight = _process_weight_wrapper
            changed.append("process_weight")

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


def _parse_to_target(args, kwargs):
    """Resolve the destination device/dtype of a ``Tensor.to`` call."""
    import torch

    torch_device = getattr(torch, "device", None)
    torch_dtype = getattr(torch, "dtype", None)
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")
    non_blocking = bool(kwargs.get("non_blocking", False))
    for arg in args:
        if isinstance(arg, torch.Tensor):
            device, dtype = arg.device, dtype or arg.dtype
        elif torch_device is not None and isinstance(arg, torch_device):
            device = arg
        elif isinstance(arg, str):
            if arg.split(":")[0] in ("cpu", "npu", "cuda", "xpu", "meta"):
                device = torch.device(arg) if torch_device is not None else arg
            else:
                dtype = getattr(torch, arg, dtype)
        elif torch_dtype is not None and isinstance(arg, torch_dtype):
            dtype = arg
    return device, dtype, non_blocking


def _is_host_weight(tensor):
    """True for a host tensor big enough to hit the A2 TransData defect.

    Parameters (``requires_grad``) keep the stock path so that module placement
    (``nn.Module.to``/``_apply``) is untouched; small transfers (ids, masks,
    quant scales below the threshold) are left alone too.
    """
    try:
        return (
            tensor.device.type == "cpu"
            and tensor.numel() >= _SAFE_TO_MIN_NUMEL
            and not getattr(tensor, "requires_grad", False)
        )
    except Exception:
        return False


def _install_safe_to():
    """Route large CPU->NPU ``Tensor.to``/``npu()`` moves through ``copy_``.

    This is the backstop for QuaRot sites this module does not know by name
    (other model families, quant post-processing, future upstream code): the
    transfer keeps identical semantics, only the op pair changes to the one
    the A2 op store supports.
    """
    global _safe_to_installed, _SAFE_TO_MIN_NUMEL
    if _safe_to_installed:
        return
    import functools
    import os

    import torch

    if os.environ.get("UCM_QUAROT_SAFE_TO", "1").lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        logger.info("UCM QuaRot safe Tensor.to disabled (UCM_QUAROT_SAFE_TO)")
        _safe_to_installed = True
        return

    override = os.environ.get("UCM_QUAROT_SAFE_TO_MIN_NUMEL")
    if override:
        try:
            _SAFE_TO_MIN_NUMEL = int(override)
        except ValueError:
            logger.warning(
                "UCM QuaRot safe Tensor.to: ignoring bad "
                "UCM_QUAROT_SAFE_TO_MIN_NUMEL=%r",
                override,
            )

    orig_to = torch.Tensor.to
    if not getattr(orig_to, "_ucm_quarot_safe_to", False):

        @functools.wraps(orig_to)
        def to(self, *args, **kwargs):
            try:
                if _is_host_weight(self):
                    device, dtype, non_blocking = _parse_to_target(args, kwargs)
                    if device is not None and device.type != "cpu":
                        host = self if dtype is None or dtype == self.dtype else self.to(dtype)
                        out = torch.empty(host.shape, dtype=host.dtype, device=device)
                        _write_host_block(out, host)
                        _log_safe_to(self, device)
                        return out
            except Exception:  # fall through to the stock implementation
                pass
            return orig_to(self, *args, **kwargs)

        to._ucm_quarot_safe_to = True
        torch.Tensor.to = to

    orig_npu = getattr(torch.Tensor, "npu", None)
    if orig_npu is not None and not getattr(orig_npu, "_ucm_quarot_safe_to", False):

        @functools.wraps(orig_npu)
        def npu(self, *args, **kwargs):
            try:
                if _is_host_weight(self):
                    device = args[0] if args else "npu"
                    out = torch.empty(self.shape, dtype=self.dtype, device=device)
                    _write_host_block(out, self)
                    _log_safe_to(self, device)
                    return out
            except Exception:  # fall through to the stock implementation
                pass
            return orig_npu(self, *args, **kwargs)

        npu._ucm_quarot_safe_to = True
        torch.Tensor.npu = npu

    _safe_to_installed = True
    logger.info(
        "UCM QuaRot safe Tensor.to installed (>= %d elements, kill switch "
        "UCM_QUAROT_SAFE_TO=0, threshold override "
        "UCM_QUAROT_SAFE_TO_MIN_NUMEL)",
        _SAFE_TO_MIN_NUMEL,
    )


def _log_safe_to(tensor, device):
    global _safe_to_logged
    if _safe_to_logged:
        return
    _safe_to_logged = True
    logger.info(
        "UCM QuaRot safe Tensor.to: routed a %s %s transfer to %s through copy_",
        tuple(tensor.shape),
        tensor.dtype,
        device,
    )


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

# Installed eagerly: `apply_all_patches()` runs right after `import vllm`, i.e.
# before any model or draft is constructed, and the net must cover paths whose
# module names this patch does not know.
try:
    _install_safe_to()
except Exception as e:  # never break startup because of the backstop
    logger.error("UCM QuaRot safe Tensor.to install failed: %s", e)
