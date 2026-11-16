"""Probe which host->device transfer forms work on this Ascend box.

Run this on the machine that fails to start (single card is enough, e.g.
``ASCEND_RT_VISIBLE_DEVICES=0 python probe_a2_h2d.py``). It answers one
question: does ``EZ3002 / TransData(groups=1)`` depend on the *destination
layout* (FRACTAL_NZ) or on the transfer size?

Every probe forces a synchronize so a failure is attributed to the right step;
failures are printed, not raised.
"""

import torch
import torch_npu  # noqa: F401  (registers the npu device)

DEV = "npu:0"
ACL_FORMAT_ND = 0
ACL_FORMAT_FRACTAL_NZ = 29
SHAPE = (4096, 4096)


def _sync():
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _fmt(t):
    try:
        return torch_npu.get_npu_format(t)
    except Exception:
        return "?"


def probe(name, fn):
    try:
        detail = fn()
        _sync()
        print(f"[ OK ] {name}" + (f"  ({detail})" if detail else ""))
    except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
        print(f"[FAIL] {name}\n       {type(exc).__name__}: {str(exc).splitlines()[0]}")


def main():
    print(f"torch={torch.__version__} npu={DEV}")
    print(f"weight_nz_mode: ", end="")
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        print(get_ascend_config().weight_nz_mode)
    except Exception:
        print("n/a")

    host_bf16 = torch.randn(*SHAPE, dtype=torch.bfloat16)
    host_fp32 = host_bf16.float()

    def nd_bf16():
        dst = torch.empty(SHAPE, dtype=torch.bfloat16, device=DEV)
        dst.copy_(host_bf16)
        return f"format={_fmt(dst)}"

    def nd_fp32_source():
        dst = torch.empty(SHAPE, dtype=torch.bfloat16, device=DEV)
        dst.copy_(host_fp32)
        return f"format={_fmt(dst)}"

    def nd_fp32_dst():
        dst = torch.empty(SHAPE, dtype=torch.float32, device=DEV)
        dst.copy_(host_fp32)
        return f"format={_fmt(dst)}"

    def to_device():
        dev = host_bf16.to(DEV)
        return f"format={_fmt(dev)}"

    def to_device_fp32():
        dev = host_fp32.to(DEV)
        return f"format={_fmt(dev)}"

    def nz_dst():
        dst = torch.empty(SHAPE, dtype=torch.bfloat16, device=DEV)
        dst.copy_(host_bf16)  # ND first, so the cast starts from real data
        nz = torch_npu.npu_format_cast(dst, ACL_FORMAT_FRACTAL_NZ)
        nz.copy_(host_bf16)
        return f"format={_fmt(nz)}"

    def nz_dst_chunked():
        dst = torch.empty(SHAPE, dtype=torch.bfloat16, device=DEV)
        dst.copy_(host_bf16)
        nz = torch_npu.npu_format_cast(dst, ACL_FORMAT_FRACTAL_NZ)
        rows = 256
        for start in range(0, SHAPE[0], rows):
            stop = min(start + rows, SHAPE[0])
            nz[start:stop].copy_(host_bf16[start:stop])
        return f"format={_fmt(nz)}"

    def nz_back_to_nd():
        dst = torch.empty(SHAPE, dtype=torch.bfloat16, device=DEV)
        dst.copy_(host_bf16)
        nz = torch_npu.npu_format_cast(dst, ACL_FORMAT_FRACTAL_NZ)
        return f"rotated back, format={_fmt(torch_npu.npu_format_cast(nz, ACL_FORMAT_ND))}"

    print()
    probe("A. copy_ bf16 host -> fresh ND bf16 device tensor", nd_bf16)
    probe("B. copy_ fp32 host -> fresh ND bf16 device tensor", nd_fp32_source)
    probe("C. copy_ fp32 host -> fresh ND fp32 device tensor", nd_fp32_dst)
    probe("D. bf16 host .to('npu')", to_device)
    probe("E. fp32 host .to('npu')", to_device_fp32)
    probe("F. copy_ bf16 host -> FRACTAL_NZ device tensor (whole)", nz_dst)
    probe("G. copy_ bf16 host -> FRACTAL_NZ device tensor (256 rows)", nz_dst_chunked)
    probe("H. FRACTAL_NZ -> ND cast", nz_back_to_nd)

    print()
    print(
        "Reading the result:\n"
        "  A-E fail           -> the box rejects large 2-D host->device transfers\n"
        "                        in general; only small/chunked copies are safe.\n"
        "  A-E ok, F fails    -> the destination layout (FRACTAL_NZ) is the trigger;\n"
        "                        the fix is to write into a fresh ND tensor and let\n"
        "                        the layer's post-load hook re-apply NZ.\n"
        "  F fails, G ok      -> NZ is fine for small blocks; chunked writes fix it.\n"
        "  everything ok      -> the failure needs the real model context; capture\n"
        "                        torch_npu.get_npu_format(layer.weight.data) right\n"
        "                        before the failing copy."
    )


if __name__ == "__main__":
    main()
