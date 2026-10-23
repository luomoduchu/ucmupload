# UCM H100 benchmark bundle

vLLM 0.27.1 + Qwen3-32B + 2 × H100 80GB (TP=2).

- [Download the scripts](ucmgraph-h100-benchmark.tar.gz)
- [完整运行说明（中文）](USAGE.md)
- [SHA-256 checksum](SHA256SUMS)

在 Linux 主机将包放入 /data/ucm-test 后执行：

```bash
cd /data/ucm-test
sha256sum -c SHA256SUMS
tar -xzf ucmgraph-h100-benchmark.tar.gz
```

随后按 USAGE.md 启动容器、安装依赖、运行 graph off/on 对照测试。
源码安装脚本会 clone 用户 UCM 仓库并固定到提交
`89909fd0c5c6fb140a5dd9e0ec0ded213e4fa6e5`。

已完成本地 Shell/Python 语法、YAML 和指标解析检查；尚未在 H100 实机完成端到端验证。
