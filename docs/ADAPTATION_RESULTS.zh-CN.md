# MiniLM LoRA 适配：完整结果与误差分析

**本轮过滤策略缓解了检索难负例组的退化，但没有建立相对未适配编码器的稳定优势。** SciFact 上，`filtered − retrieved` 的 nDCG@10 差值为 `+0.151993`，配对查询 bootstrap 95% 区间 `[+0.115510, +0.189523]`；`filtered − base` 仅为 `+0.015657`，区间跨 0。NFCorpus 上，过滤组也明显高于检索难负例组，但仍低于未适配 baseline，差值为 `−0.014376`，区间 `[−0.022987, −0.005683]`。这一结果支持继续研究负例构造与适配退化，不支持“过滤后全面超过预训练模型”或“已经证明去噪有效”的表述。

本报告对应完整的两数据集、四编码器实验。原始证据：[固定协议](../results/adaptation/protocol.json)、[训练报告](../results/adaptation/training/training_report.json)、[检索报告](../results/adaptation/report.json)、[独立审计](../results/adaptation/independent_audit.json)、[逐查询分析](../results/adaptation/analysis.json)。数值表保留四位小数便于阅读；差值与区间保留六位，完整精度以 JSON 为准。

![MiniLM 四编码器质量及主比较区间](../results/adaptation/comparison.svg)

图中前两面板使用相同的 0–1 纵轴，第三面板显示预先指定的主比较。误差条来自查询 bootstrap，不能解释为多次训练的方差。[PNG 版本](../results/adaptation/comparison.png) · [绘图脚本](../scripts/plot_adaptation_results.py)。

## 实验实际训练了什么

基础模型为官方 `sentence-transformers/all-MiniLM-L6-v2` 浮点权重，使用 `transformers.BertModel`、attention-mask mean pooling 和 L2 归一化。仅在最后一个 Transformer block 的 attention Q/V 上添加 `rank=8, alpha=16` 的 LoRA，共 **12,288 个可训练参数**。三个训练组均完成 **648 次更新、648 次非零梯度步骤**；保存了 adapter 权重、参数变化量、训练轨迹与可恢复 checkpoint。它是实际的检索编码器参数适配，不是生成式大模型指令微调。

| 固定条件 | 本轮配置 |
|---|---|
| 内部划分 | 809 条官方训练查询按共享相关文档/规范化 claim 分组，647 train / 162 validation |
| 训练预算 | 每组 2 epochs，batch size 2，648 次更新、1,294 次查询曝光 |
| 损失 | 每条查询一个标注正例、两个显式负例的 InfoNCE；无 in-batch negatives |
| 优化 | AdamW，学习率 0.001，weight decay 0.01，温度 0.05，梯度裁剪 1.0，seed 42 |
| 模型选择 | 预先固定最后一个 epoch；验证 loss 仅监测，不早停，不按 test 挑组，不用全部 809 条查询重训 |
| 评测对照 | 同一浮点基础模型、128-token 上限、完整语料精确余弦排名 |
| 本地资源 | i5-1130G7 / 16GB 共享电脑，PyTorch CPU 两线程；三组顺序执行，无付费或外部模型计算 |

`random` 随机选择未标为相关的文档；`retrieved` 选择冻结 RRF top-50 中最靠前的非正例；`filtered` 在同一候选池中先做训练家族保护，再用冻结 MiniLM 的相似度风险规则过滤，不足时随机补足。三个组共享训练查询、正例、初始化、查询顺序和更新预算，**没有匹配负例难度、token 总数或运行时间**。完整定义及教师边界见[实验协议](ADAPTATION_PROTOCOL.zh-CN.md)。

## 完整检索指标

每个模型都让测试查询对全部文档排序，输出 top-50 TREC 排名；没有向测试结果补入标注正例。下表是密集检索指标，不能归到适配器的 RRF 推理模式。

### SciFact：5,183 篇文档、300 条测试查询

| 编码器 | nDCG@10 | MRR@10 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|
| `base` | 0.6239 | 0.5784 | 0.7743 | 0.8853 |
| `random` | 0.6291 | 0.5859 | 0.7832 | 0.8827 |
| `retrieved` | 0.4875 | 0.4617 | 0.5943 | 0.7802 |
| `filtered` | 0.6395 | 0.6048 | 0.7666 | 0.8797 |

过滤组的 nDCG/MRR 点估计高于 baseline，但 Recall@10/50 更低，因此连同一数据集上的四项指标也没有一致提升。检索难负例组的退化同时涉及前排质量与召回集合，不能只归为 top-10 内排序波动。[保存的全部排名](../results/adaptation/scifact/runs/)。

### NFCorpus：3,633 篇文档、323 条测试查询

| 编码器 | nDCG@10 | MRR@10 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|
| `base` | 0.3149 | 0.5123 | 0.1537 | 0.2521 |
| `random` | 0.3052 | 0.4855 | 0.1475 | 0.2488 |
| `retrieved` | 0.1891 | 0.3388 | 0.0810 | 0.1334 |
| `filtered` | 0.3005 | 0.4969 | 0.1417 | 0.2404 |

三个适配组的四项指标均低于未适配 baseline。模型在 SciFact 上固定后直接用于 NFCorpus，没有用后者的 train/dev/test 标签更新参数、阈值或模型选择。NFCorpus 是非技术健康问题到医学文献的检索，和 SciFact 有任务及查询分布差异，但仍共享生物医学主题。其 1/2 两级相关度保留到 nDCG，采用 TREC 口径的**线性 gain `rel`**；MRR/Recall 按 `rel > 0` 计算。不能把结果推广为完全无关领域或医疗事实核验能力。[保存的全部排名](../results/adaptation/nfcorpus/runs/)。

## 预先指定的主比较与全部次比较

以下区间均直接来自主实验 `report.json` 的 NumPy 配对查询 bootstrap，2,000 次重采样，seed 42。差值是绝对 nDCG 单位，不是相对百分比；主比较固定为 `filtered − retrieved`，其余三项为相对同配置 baseline 的次比较。

| 数据集 | 比较 | nDCG@10 差值 | 95% 区间 |
|---|---|---:|---|
| SciFact | **filtered − retrieved（主）** | **+0.151993** | **[+0.115510, +0.189523]** |
| SciFact | random − base | +0.005203 | [−0.013429, +0.024154] |
| SciFact | retrieved − base | −0.136336 | [−0.178151, −0.095668] |
| SciFact | filtered − base | +0.015657 | [−0.010680, +0.042043] |
| NFCorpus | **filtered − retrieved（主）** | **+0.111347** | **[+0.092410, +0.131318]** |
| NFCorpus | random − base | −0.009666 | [−0.017041, −0.002223] |
| NFCorpus | retrieved − base | −0.125723 | [−0.147112, −0.105592] |
| NFCorpus | filtered − base | −0.014376 | [−0.022987, −0.005683] |

按逐查询 nDCG 比较，过滤组对检索难负例组在 SciFact 上有 **103 胜 / 171 平 / 26 负**，在 NFCorpus 上有 **168 胜 / 113 平 / 42 负**。这些计数描述本次测试查询，并不说明所有查询都受益。独立分析脚本使用标准库随机数重新采样，其完整测试区间端点与主报告略有不同；本节不混用两种区间来源。

## 采样审计：过滤改变了什么

[负例审计](../results/adaptation/negative_mining_audit.json)读取训练前固定的采样计划，覆盖 647 条训练查询的 **31,675 个候选查询文档对**。教师是冻结的同源 MiniLM INT8、256-token 编码器，按 `s(候选) ≥ s(选中正例) − 0.05` 标记相似度风险；它不是经过独立标注验证的假负例分类器。

| 可直接检查的事实 | 结果 |
|---|---:|
| 家族保护标记的候选对 | 32 |
| 教师相似度标记的候选对 | 6,946 |
| 同时具有两类标记的候选对 | 19 |
| 相比 retrieved 改变负例集合的查询 | 391 / 647（60.43%） |
| 实际替换的固定负例槽位 | 673 / 1,294（52.01%） |
| filtered 随机补足负例 | 26 |
| 未训练的 family-only 影子规则替换量 | 11 / 1,294（0.85%） |
| 未训练的 teacher-only 影子规则替换量 | 670 / 1,294（51.78%） |

对有保存教师分数的选中负例，`retrieved` 的平均相似度为 `0.5235`、平均原始 RRF rank 为 `2.1414`；`filtered` 分别为 `0.4438` 和 `7.7658`。后者统计覆盖 1,268 个池内负例，不含 26 个随机补足项。这说明规则同时移除了更靠前、更相似的候选，**难度变化是不能排除的解释**。随机组只有 6 / 1,294 个负例落入已评分候选池，其余分数缺失，不能用这 6 个样本代表随机组整体难度。

两个单独规则的影子集合没有训练对应模型，不能当作质量消融。审计也没有独立相关性标签；masked-positive audit 明确为 `not_run`。因此不能把 6,946 个标记写成“发现的错误标签”，不能把 52.01% 的替换率写成去噪准确率，也不能从主比较单独识别家族保护与相似度过滤各自的贡献。

## 验证损失与检索质量给出了不同信号

三组起始验证 InfoNCE 相同，均为 `1.218033`。下表的验证负例固定为检索负例，最终权重按预先固定的 epoch 选择。

| 组别 | 最终验证 InfoNCE | SciFact nDCG@10 | NFCorpus nDCG@10 | LoRA 参数变化 L2 |
|---|---:|---:|---:|---:|
| `random` | 1.355410 | 0.6291 | 0.3052 | 3.244746 |
| `retrieved` | 1.184370 | 0.4875 | 0.1891 | 3.952478 |
| `filtered` | 1.368110 | 0.6395 | 0.3005 | 3.743834 |

检索难负例组的验证 loss 下降，却在两个全库检索测试中退化最多；过滤组验证 loss 上升，却优于该组的检索质量。这个反例说明固定两负例上的三分类监测与完整语料排序不是同一问题。过滤是否让训练任务变简单、困难负例是否提供了有偏监督、不同采样与固定学习率是否交互，都只能作为后续假设。这里没有根据测试分数追加调参，也没有通过 loss 选择一个测试赢家。

## 294-query 事后敏感性分析

SciFact 官方训练集（包含内部验证部分）与测试集有 6 条规范化 claim 重叠，测试 ID 为 `1292、13、1359、279、814、870`。规范化会将数字变体合并，不等于语义完全相同。主结果保留全部 300 条查询；另从保存排名中剔除这 6 条，只作为事后诊断，没有重新训练。

| 编码器 | 保留 294 条查询的 nDCG@10 |
|---|---:|
| `base` | 0.622986 |
| `random` | 0.628295 |
| `retrieved` | 0.485804 |
| `filtered` | 0.636955 |

| 比较 | 差值 | 事后配对 bootstrap 95% 区间 |
|---|---:|---|
| filtered − retrieved | +0.151151 | [+0.113170, +0.189304] |
| random − base | +0.005309 | [−0.014549, +0.025483] |
| retrieved − base | −0.137182 | [−0.178670, −0.095680] |
| filtered − base | +0.013969 | [−0.013181, +0.041355] |

方向与完整测试一致：过滤缓解检索难负例组退化，过滤对 baseline 的区间仍跨 0。本节区间来自 `analysis.json` 的标准库 `random.Random.choices`，2,000 次重采样、seed 42。这项检查只移除规范化后完全相同的 claim，不排除语义改写、共享文献或上游预训练暴露，不能作为“完全无泄漏”的证据。

## 两个真实极端案例

以下案例按 `filtered − retrieved` 的逐查询 nDCG 变化事后选取一个最大改善和一个最大退化，仅用于解释可观察错误，不能代替总体结果。查询、文档标题、原始摘要和保存的 TREC 排名均在本地核对；未标注文献称为“未标为相关”，不自动判断为真实负例。

**改善案例：SciFact query `1062`，目标文档 `20381484`。** 查询涉及 S-nitrosylated GAPDH 对组蛋白去乙酰化酶的转亚硝基化作用。目标摘要《GAPDH Mediates Nitrosylation of Nuclear Proteins》明确讨论 SNO-GAPDH 对包括 SIRT1、HDAC2 在内的核蛋白的作用。该文献在 `base/random/retrieved/filtered` 中分别位于第 **1/3/26/1** 位；`filtered − retrieved` 的 nDCG@10 为 `+1.000000`。检索难负例组的第 1 篇 `32985041` 讨论较宽泛的蛋白 S-nitrosylation 和 nNOS 信号，具有相近主题，但不是该查询的标注目标。可观察到的是目标文献从 top-10 外返回首位；不能据此认定某个训练负例曾被错标或过滤修复了该条事实。

**退化案例：SciFact query `729`，目标文档 `26851674`。** 查询涉及缺失 SHP-2 MAPK 通路的 knockin 小鼠出现淋巴结肿大。目标摘要《Dissection of signaling cascades through gp130 in vivo: reciprocal roles for STAT3- and SHP2-mediated signals in immune responses》描述了 SHP2 信号缺陷小鼠与淋巴结肿大等表型。该文献在 `base/random/retrieved` 中均为第 **1** 位，在 `filtered` 中降至第 **10** 位；nDCG@10 从 `1.000000` 降至 `0.289065`，差值 `−0.710935`。过滤组第 1 篇 `7521113` 研究单核细胞和组织巨噬细胞的命运追踪；主题同属免疫细胞研究，但该查询未将其标为相关。这里 Recall@10 仍为 1，下降发生在前排次序，说明过滤策略也可能损害具体机制和表型的匹配。

完整案例列表及前十篇文档见[逐查询分析](../results/adaptation/analysis.json)，原始排名可在 SciFact 的 [retrieved](../results/adaptation/scifact/runs/retrieved.trec) 与 [filtered](../results/adaptation/scifact/runs/filtered.trec) 文件核对。

## 工程验证、运行成本与复现

[独立审计](../results/adaptation/independent_audit.json)为 `ok: true`，从 TREC 排名重新计算 **32 个汇总指标**及全部逐查询指标，检查配对均值、完整查询覆盖、训练计划/协议身份、相同初始化、adapter/run 哈希与声明的训练完成状态。它不重放优化器，不重做模型推理，也不验证 bootstrap 端点或耗时。审计通过支持产物的一致性，不消除实验设计本身的局限。

本轮 **97 项本地测试通过**。[真实适配器 API smoke 记录](../results/adaptation/service_smoke.json)还检查了三条查询的 dense top-5 与评测排名完全一致、一次 RRF 请求、错误输入、不兼容 v1 重排器的 409 响应，以及模型/缓存身份；真实编码了查询，没有重新编码语料。这是 FastAPI TestClient 内的顺序功能检查，没有启动实际 HTTP 服务，也不是并发负载或线上性能测试。

训练中发生过断点恢复，历史见[执行记录](../results/adaptation/execution_notes.json)。训练报告的 `elapsed_seconds=734.954621` 仅覆盖**最后这次调用**；其中随机组从第 648 步 checkpoint 恢复，之前的优化耗时没有包含在内。该值不能写成“三组从零训练总共约 12 分钟”，也不能用来比较采样策略速度。评测报告的 `evaluation_seconds=658.345702` 同样只属于该次评测调用，包含缓存路径的运行范围；点积排序时间不是端到端在线查询延迟。本报告不据此声称总训练成本、QPS 或生产收益。

复现训练和完整评测见[CPU 操作指南](CPU_ADAPTATION.zh-CN.md)。在已有产物上运行独立检查与画图：

```bash
python scripts/verify_adaptation_results.py --output results/adaptation/independent_audit.json
python scripts/analyze_adaptation_results.py --output results/adaptation/analysis.json
python scripts/plot_adaptation_results.py
```

绘图可选安装 `matplotlib`，它不属于模型训练的核心依赖。脚本要求两数据集结果完整且独立审计通过，避免将部分评测画成最终结果。

## 结论边界

- 只有一个优化 seed；查询 bootstrap 不估计训练随机性，未按 claim 家族聚类，也未作多重比较校正。
- 过滤是家族保护与相似度阈值的联合处理，同时改变难度与随机补足比例。没有独立假负例真值，没有匹配难度的训练对照，不能作去噪机制的单独因果归因。
- 数据规模有限，内部训练仅 647 条查询。NFCorpus 检查的是存在主题重叠的任务/查询分布迁移；其数据许可限于原作者允许的学术用途。
- 128-token 截断、单个学习率、最后层 Q/V LoRA 和两轮预算限定了结论，不能把检索难负例组的失败推广为难负例训练普遍无效。
- v1 的 INT8、256-token 编码器及 LambdaMART 管线与本轮浮点、128-token dense 对照不同，而且最终训练查询数也不同；v1 的 `0.7149` 不能作为受控的 LoRA 前后对照。[v1 报告](RESULTS.zh-CN.md)应独立阅读。
- Qwen3-Embedding-0.6B Q8 是另一项冻结模型基线，固定资产、输入与完整质量报告入口见 [Qwen 基线文档](MODERN_BASELINE.zh-CN.md)。本轮三组训练没有使用 Qwen 教师，也没有训练或蒸馏 Qwen 权重。
