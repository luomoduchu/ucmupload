"""Reject runs that did not exercise UCM backend loads or had request errors."""

import json
import re
import sys
from pathlib import Path


def counters(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^(ucm:[\w:]+)(?:\{.*\})?\s+([\d.eE+-]+)(?:\s|$)", line)
        if match:
            name, value = match.groups()
            result[name] = result.get(name, 0.0) + float(value)
    return result


def main():
    mode, before_path, after_path, result_path = sys.argv[1:]
    if mode not in ("on", "off"):
        raise SystemExit("Mode must be on or off")
    before, after = counters(before_path), counters(after_path)
    delta = {name: value - before.get(name, 0) for name, value in after.items()}
    assert all(
        v >= 0 for k, v in delta.items() if k.endswith("_total")
    ), "Counters reset during benchmark; check whether workers restarted"
    backend_metric = (
        "ucm:cache_transfer_graph_backend_shards_total"
        if mode == "on"
        else "ucm:cache_load_backend_shards_total"
    )
    if backend_metric not in after:
        raise SystemExit(
            f"Required metric not exported in after snapshot: {backend_metric}. "
            "Check enable_metrics, worker metric export and the UCM connector. "
            f"Snapshot: {after_path}"
        )
    backend = delta[backend_metric]
    batches = delta.get("ucm:cache_transfer_graph_backend_batches_total", 0)
    shards = delta.get("ucm:cache_transfer_graph_backend_shards_total", 0)
    result = json.loads(Path(result_path).read_text())
    assert result["completed"] == result["num_prompts"], "Benchmark requests failed"
    if backend <= 0:
        raise SystemExit(
            f"Metric exported but no backend load increment: {backend_metric}; "
            f"before={before.get(backend_metric, 0)}, after={after[backend_metric]}. "
            "Check cache hits, completed seed writes and metric export timing."
        )
    if mode == "on":
        assert batches > 0 and shards > 0, "Graph path was not exercised"
    else:
        assert (
            batches == 0 and shards == 0
        ), "Expected graph OFF, but graph counters increased"
    errors = {
        k: v
        for k, v in delta.items()
        if v > 0
        and (
            "errors_total" in k or "failed_shards_total" in k or "queue_full_total" in k
        )
    }
    assert not errors, f"UCM errors: {errors}"
    print(
        json.dumps(
            {
                "backend_metric": backend_metric,
                "backend_shards": backend,
                "graph_batches": batches,
                "graph_shards": shards,
                "graph_shards_per_batch": shards / batches if batches else None,
            },
            indent=2,
        )
    )
    print("Load path checks passed. Compare hit coverage before interpreting timings.")


if __name__ == "__main__":
    main()
