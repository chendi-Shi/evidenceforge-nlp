# EvidenceForge｜科学文献检索、负例采样与神经适配

[GitHub 项目](https://github.com/chendi-Shi/evidenceforge-nlp) · [神经适配结果](ADAPTATION_RESULTS.zh-CN.md) · [混合检索结果](RESULTS.zh-CN.md)

- 在普通电脑上完成 MiniLM 最后一层 Q/V 的 rank-8 LoRA 训练，更新 12,288 个参数；固定 647/162 组级训练验证划分，以相同初始化和每组 648 步预算对比随机负例、检索难负例与风险过滤负例，在完整 SciFact（5,183 文档／300 查询）及 NFCorpus（3,633 文档／323 查询）上评测。
- 通过配对 bootstrap、逐查询分析及去重叠敏感性检查，发现并量化难负例训练退化：过滤组相对未过滤组的 nDCG@10 差值为 SciFact +0.1520、NFCorpus +0.1113；同时披露其相对未适配基线的 SciFact 差值区间跨 0、NFCorpus 仍下降，避免把采样对照收益误写为泛化提升。
- 构建 BM25、MiniLM INT8、RRF 与 LambdaMART 混合检索基线；在 SciFact 300 查询上，监督重排使 nDCG@10 从 0.6870 升至 0.7149（相对 +4.06%，差值 95% 区间 [0.0117, 0.0445]），并完成相同候选池上的重排特征消融。
- 在本机完成 Qwen3-Embedding-0.6B Q8 冻结基线的完整 SciFact 评测（nDCG@10 0.6752，128-token 配置）；实现适配器 FastAPI 接口、断点恢复与独立指标复算，本地 97 项测试通过，真实模型 API 排名与评测一致，公开三组适配器及复现记录。[Qwen 条件与结果](MODERN_BASELINE.zh-CN.md)。

简历空间有限时，可保留第 1、3、4 条，并在面试中完整解释第 2 条的负结果。v1 的 +4.06% 属于 LambdaMART 重排；它不是 LoRA 或 Qwen 的训练收益。MiniLM LoRA 是编码器适配，不是大语言模型指令微调；Qwen 权重未训练。以上表述只覆盖已经完成的工作及明确标注的对照角色，应在能独立运行和解释后使用。
