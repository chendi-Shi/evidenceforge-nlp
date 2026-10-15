# EvidenceForge｜检索蒸馏、表示保持与迁移评测

[GitHub 项目](https://github.com/chendi-Shi/evidenceforge-nlp) · [v3 完整结果](DISTILLATION_RESULTS.zh-CN.md) · [运行与证据说明](DISTILLATION.zh-CN.md)

以下三条以已完成的 v3 为主线；使用前应能独立运行、定位实现并解释实验取舍。

- 构建 Qwen3-Embedding-0.6B Q8 → MiniLM 的离线检索蒸馏流程，冻结前四层、训练后两层 3,548,928 个参数；固定 MS MARCO 来源的 12,000/1,000 训练验证查询，对监督 CE、教师 KL、原始表示保持及组合目标执行四组 × 三种子对照，共完成 9,000 次有效优化器更新。
- 仅按源域验证选模，在完整 SCIDOCS（25,657 文档、1,000 查询）评测：KD+表示保持的 nDCG@10 三种子均值为 0.215630，较 KD 的 0.209372 相对提高 2.99%，各种子的配对查询 95% 区间均为正；同时披露所有种子仍低于冻结 base 0.216406，将结果界定为缓解负迁移。
- 实现冻结前缀缓存、长度分桶前向、完整候选池损失和优化器断点恢复；保存协议、训练轨迹、选中/最终权重身份与逐查询排名，通过独立脚本复算 13 份排名的四项检索指标及主比较区间，并分析源域收益、表示漂移和四组交互量。

三种子改变的是批次顺序，不能写成覆盖全部训练随机性；查询区间条件于各自已训练模型。项目没有训练 Qwen 权重，没有开展大语言模型指令微调，也没有证明超过原模型、SOTA 或新算法。独立审计核对保存产物，不重新推理或认证训练历史。数字对应 [主报告](../results/distillation/report.json)、[次要分析](../results/distillation/analysis.json)与[审计](../results/distillation/independent_audit.json)。

## 历史工作，可按岗位替换一条

- **v1，检索与排序工程：** 构建 BM25、MiniLM INT8、RRF 与 LambdaMART 基线；在 SciFact 300 查询上，监督重排使 nDCG@10 从 0.6870 升至 0.7149（相对 +4.06%，配对差值 95% 区间 [0.0117, 0.0445]），完成相同候选池上的重排特征消融与 FastAPI 接口。[v1 结果](RESULTS.zh-CN.md)
- **v2，神经适配与负例分析：** 训练 12,288 参数的 MiniLM Q/V LoRA，以相同初始化和每组 648 步预算比较三种负例采样；过滤组较检索难负例组的 nDCG@10 差值为 SciFact +0.1520、NFCorpus +0.1113，同时报告其相对未适配基线未建立稳定优势，并完成 Qwen Q8 的 SciFact 冻结基线评测（0.6752，128 tokens）。[v2 结果](ADAPTATION_RESULTS.zh-CN.md) · [Qwen 基线条件](MODERN_BASELINE.zh-CN.md)

历史数字不能归为 v3 蒸馏收益。v1 使用 ONNX INT8/256 tokens，v2 的适配公平对照使用浮点 MiniLM/128 tokens，v3 使用浮点 MiniLM/256 tokens；数据与配置不同，不跨表计算训练提升。当前项目是公开基准上的算法实验与本地复现系统，不写未经测量的线上 QPS、用户量或生产收益。
