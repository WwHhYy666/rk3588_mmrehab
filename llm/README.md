# Local language-model adapter

`llm/rkllm_proxy.py` 将 Rockchip 官方 Qwen RKLLM Flask server 的接口适配为康复应用使用的 `/health` 与 `/generate`。

Qwen 权重统一放在 `models/language/qwen/`；官方 server 和 `librkllmrt.so` 属于板端运行环境，不提交到仓库。
