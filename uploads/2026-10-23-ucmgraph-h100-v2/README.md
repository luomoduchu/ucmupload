# UCM H100 benchmark bundle v2

修复 graph on 误检查 legacy 计数的问题；只需替换 [check_metrics.py](check_metrics.py)，无需重编译 UCM。
已通过 10 项本地回归检查，详见 [test_check_metrics.py](test_check_metrics.py)。旧版压缩包保留在原目录。

vLLM 0.27.1 + Qwen3-32B + 2 × H100 80GB (TP=2).

- [Download the scripts](ucmgraph-h100-benchmark-v2.tar.gz)
- [完整运行说明（中文）](USAGE.md)
- [SHA-256 checksum](SHA256SUMS)

在 Linux 主机将包放入 /data/ucm-test 后执行：

```bash
cd /data/ucm-test
sha256sum -c SHA256SUMS
tar -xzf ucmgraph-h100-benchmark-v2.tar.gz
```

随后按 USAGE.md 启动容器、安装依赖、运行 graph off/on 对照测试。
源码安装脚本会 clone 用户 UCM 仓库并固定到提交
`89909fd0c5c6fb140a5dd9e0ec0ded213e4fa6e5`。

已完成本地 Shell/Python 语法、YAML 和指标解析检查；尚未在 H100 实机完成端到端验证。
