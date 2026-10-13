# EvidenceForge｜科学文献混合检索与学习排序

[GitHub 项目](https://github.com/chendi-Shi/evidenceforge-nlp) · [实测报告与案例](RESULTS.zh-CN.md)

- 基于 BEIR SciFact 的 5,183 篇论文摘要构建 BM25、冻结 MiniLM（ONNX INT8）与 RRF 检索管线；从 top 50 候选挖掘难负例，利用 16 维检索及文本匹配特征训练 LambdaMART，809 条训练查询形成 40,512 个查询文档对。
- 采用共享相关文档与规范化论断的组级内部验证，完成五组对照及重排特征消融；在 300 条公开测试查询上，nDCG@10 从 RRF 的 0.6870 提升至 0.7149（相对 +4.06%），配对 bootstrap 增益 95% 区间为 [0.0117, 0.0445]，Recall@10 从 0.8182 提升至 0.8549，并披露官方划分的 6 个规范化论断重叠。
- 实现本地 FastAPI 检索接口、固定模型版本与缓存校验、逐查询误差分析和独立指标复核；37 个本地测试及 GitHub Linux Python 3.11/3.12 离线 CI 通过，保留模型、配置、TREC 排名和复现实验记录。
