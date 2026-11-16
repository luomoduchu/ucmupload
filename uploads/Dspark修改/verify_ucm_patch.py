"""Verify the UCM `dspark_quarot_host_align_patch` monkeypatch on the real box.

Run on the NPU host with UCM and vllm-ascend importable:

    python verify_ucm_patch.py            # npu when available, else cpu
    python verify_ucm_patch.py cpu        # force cpu

Checks, in the same order production sees them:

1. importing the UCM patch module, then importing
   `vllm_ascend.models.qwen3_dspark`, leaves the *patched* function bound on
   both `llama_eagle3` and `qwen3_dspark` (the second one is the binding that
   `align_draft_weights` actually calls);
2. the patched function still produces `target_shard.float() @ rotation.T`, with
   the unused tail rows zeroed, for both `rotation is not None` and `None`;
3. the chunked loop is exercised (chunk size forced to 2 rows).

The fake layer's weight lives on `--device`, so a passing run also proves the
remaining host->device copy of the finished block is accepted by CANN.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

import ucm.integration.vllm.patch.dspark_quarot_host_align_patch as ucm_patch

import vllm_ascend.models.llama_eagle3 as eagle3
import vllm_ascend.models.qwen3_dspark as qwen3_dspark

WEIGHT_NAME = "lm_head.weight"


def make_layer(rows: int, cols: int, *, device: str, dtype: torch.dtype) -> nn.Module:
    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.full((rows, cols), 99.0, device=device, dtype=dtype)
    )
    layer.shard_indices = None  # use the whole shard: [0, rows)
    return layer


def write_shard(root: Path, weight: torch.Tensor) -> None:
    shard = "model-00001-of-00001.safetensors"
    save_file({WEIGHT_NAME: weight}, root / shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {WEIGHT_NAME: shard}}), encoding="utf-8"
    )


def check(tag: str, got: torch.Tensor, want: torch.Tensor) -> None:
    got = got.float().cpu()
    want = want.float()
    if not torch.allclose(got, want, atol=1e-5, rtol=1e-5):
        raise AssertionError(f"{tag}: mismatch\n got={got}\n want={want}")
    print(f"  ok  {tag}")


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "npu"
    if device.startswith("npu") and not torch.npu.is_available():
        print("npu not available, falling back to cpu")
        device = "cpu"

    print("1) patch landed on the modules that use it")
    for mod in (eagle3, qwen3_dspark):
        fn = getattr(mod.load_quarot_target_layer, "_ucm_quarot_host_align", False)
        if not fn:
            raise AssertionError(f"{mod.__name__}.load_quarot_target_layer is NOT patched")
        print(f"  ok  {mod.__name__}.load_quarot_target_layer -> host alignment")

    if not getattr(qwen3_dspark.align_draft_weights, "_ucm_quarot_host_align", False):
        raise AssertionError(
            "vllm_ascend.models.qwen3_dspark.align_draft_weights is NOT patched "
            "(projection rotation would still go through Tensor.to(device))"
        )
    print("  ok  vllm_ascend.models.qwen3_dspark.align_draft_weights -> host alignment")

    ucm_patch._ALIGN_CHUNK_ROWS = 2  # force several loop iterations

    torch.manual_seed(0)
    rows, cols, tail = 5, 4, 2
    target = torch.randn(rows, cols, dtype=torch.float32)
    rotation = torch.randn(cols, cols, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_shard(root, target)

        print(f"2) numerics on device={device}")
        for dtype in (torch.float32, torch.bfloat16):
            layer = make_layer(rows + tail, cols, device=device, dtype=dtype)
            qwen3_dspark.load_quarot_target_layer(
                layer, root, (WEIGHT_NAME,), rotation, "test lm_head"
            )
            check(f"{dtype} rotated rows", layer.weight.data[:rows], target @ rotation.T)
            check(f"{dtype} tail zeroed", layer.weight.data[rows:], torch.zeros(tail, cols))

        layer = make_layer(rows + tail, cols, device=device, dtype=torch.bfloat16)
        qwen3_dspark.load_quarot_target_layer(
            layer, root, (WEIGHT_NAME,), None, "test lm_head"
        )
        check("no-rotation rows", layer.weight.data[:rows], target)
        check("no-rotation tail zeroed", layer.weight.data[rows:], torch.zeros(tail, cols))

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
