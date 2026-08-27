# AlloyModelForge SRO 与受约束化学无序算法研究笔记

本文档用于后续研究 AlloyModelForge 的 SRO/SQS 算法。它区分三个层次：

1. SRO、相关函数和 SQS 的物理含义；
2. AlloyModelForge 当前代码实际实现了什么；
3. icet、mcsqs、sqsgenerator 的方法差异，以及可以形成论文创新的方向。

本文描述对应 AlloyModelForge `0.2.0` 当前源码。后续修改算法时，应同步更新本文、科学验证测试和 benchmark。

## 1. 从随机合金到 SRO

合金的整体成分只说明各种元素的数量。例如 NiCo 等原子比合金满足

\[
c_{\mathrm{Ni}}=c_{\mathrm{Co}}=0.5.
\]

但相同的整体成分可以对应完全不同的局部排列：

- Ni 和 Co 完全随机混合；
- Ni 更喜欢与 Co 相邻，表现出异类原子有序；
- Ni 更喜欢与 Ni 相邻，表现出同类原子团簇或偏聚。

SRO（short-range order，短程有序）描述的正是有限邻近范围内的元素排列。常用的方向性 Warren–Cowley 参数为

\[
\alpha_{A\rightarrow B}^{(n)}
=1-\frac{P_{A\rightarrow B}^{(n)}}{c_B},
\]

其中 (P_{A\rightarrow B}^{(n)}) 是以 A 为中心时，第 (n) 近邻壳层出现 B 的条件概率，(c_B) 是 B 的整体浓度。

- \(\alpha=0\)：局部概率等于随机合金概率；
- \(\alpha<0\)：A 周围的 B 多于随机值，A–B 异类近邻受到偏好；
- \(\alpha>0\)：A 周围的 B 少于随机值，A–B 互相回避。

因此“降低 SRO”不是严格的科学表述。正确目标应是：

\[
\min \left|\alpha-\alpha^{\mathrm{target}}\right|.
\]

完全随机固溶体通常采用 \(\alpha^{\mathrm{target}}=0\)，而有序、偏聚或实验标定结构可以采用非零目标。

## 2. 二体相关与高阶相关

二体（pair）相关只统计元素原子对，例如第一近邻 Ni–Co、第二近邻 Ni–Ni。对于第 (n) 个壳层，可以用实际原子对数量与随机目标原子对数量之差来评价结构。

二体统计不能唯一确定结构。两个构型可能具有相同的 A–B 原子对数量，却具有不同的三原子或四原子排列。因此簇展开语言还包括：

- 一体簇：成分；
- 二体簇：pair；
- 三体簇：triplet；
- 四体簇：quadruplet；
- 更高阶簇。

经典 SQS 的思想不是让原子“看起来杂乱”，而是选择一个有限周期超胞，使其前若干个重要簇相关函数尽可能接近无限随机合金。Zunger 等人在原始 SQS 工作中指出，小型精心选择的周期结构可以比同尺寸的直接随机占位更好地复现随机合金的前几个重要相关函数。

## 3. SQS 与普通随机占位的区别

普通随机占位从满足成分的所有构型中抽取一个样本。对于有限原子数，这个样本通常存在明显的相关函数涨落，并且周期边界会引入额外的人工相关。

SQS 则求解一个离散优化问题：

\[
\sigma^*=\arg\min_{\sigma\in\Omega}
D\left(\Gamma(\sigma),\Gamma^{\mathrm{target}}\right),
\]

其中 \(\sigma\) 是晶格位点上的元素排列，\(\Omega\) 是满足成分和位点约束的构型集合，\(\Gamma\) 是选择的相关函数向量。

重要结论：

- SQS 仍是一个周期结构，不等于真正无限随机体系；
- 有限尺寸下不是所有目标相关向量都可实现；
- SQS 的质量依赖超胞尺寸、形状、所选簇、权重和搜索强度；
- 目标函数不同的软件，其内部 objective 数值不能直接横向比较；
- 横向比较必须使用同一个独立评价器，例如 AlloyModelForge 的统一 SRO 报告。

## 4. AlloyModelForge 当前 native SQS 的数学目标

当前实现位于 `src/alloyforge/chemistry.py`。它是“按相独立、固定几何、二体相关、模拟退火交换”的优化器。

### 4.1 近邻壳层

`_shell_edges` 根据相结构和晶格参数给出 FCC、BCC、HCP 的理想壳层距离，再在周期盒中建立每个壳层的无向近邻边集合

\[
E_n=\{(i,j)\mid r_{ij}\text{ 属于第 }n\text{ 壳层}\}.
\]

FCC 与 BCC 当前最多定义五层理想距离；HCP 由 (a,c) 的若干组合距离排序得到。构建器当前使用正交周期盒，因此其 `cKDTree(boxsize=...)` 最小镜像处理与当前几何一致。未来若允许一般三斜晶胞，这一实现必须改为通用晶胞的最小镜像算法。

### 4.2 随机目标原子对数

对于无向原子对，当前 `_counts_and_targets` 采用

\[
T_{AA}^{(n)}=|E_n|c_A^2,
\]

\[
T_{AB}^{(n)}=|E_n|2c_Ac_B,\quad A\ne B.
\]

这里的因子 2 来自无向混合原子对 A–B 与 B–A 的合并。所有目标对数的总和等于该壳层边数。

### 4.3 当前目标函数

每个壳层采用近似 Pearson 型误差：

\[
E_n=\sum_{A\le B}
\frac{\left(N_{AB}^{(n)}-T_{AB}^{(n)}\right)^2}
{\max(T_{AB}^{(n)},1)}.
\]

总优化目标为

\[
E=\sum_n w_nE_n,\qquad w_n=\frac{1}{n}.
\]

注意：报告中的 `total_objective` 是各壳层未加权 objective 之和，而模拟退火内部使用 (1/n) 加权值。两者用途不同，未来论文 benchmark 应同时明确报告“优化目标”和“统一审计指标”。

### 4.4 精确成分守恒

初始元素分配阶段已经把每相的小数成分转化为整数原子数。native SQS 只在同一相内选择两个不同元素并交换，因此：

- 原子位置不变；
- 每相元素总数严格不变；
- 全局元素总数严格不变；
- 晶格、晶粒和相标签不变。

这比“随机把某个位点重新赋成一种元素”更安全，因为后者可能造成成分漂移。

### 4.5 增量更新

程序预先为每个原子建立壳层边的邻接表。交换位点 (i,j) 后，只有与这两个位点相连的边可能改变。因此无需重新统计整个模型，只计算受影响边的计数变化

\[
\Delta\mathbf N_n
=\mathbf N_{n,\mathrm{new}}-\mathbf N_{n,\mathrm{old}},
\]

并求

\[
\Delta E=E(\mathbf N+\Delta\mathbf N)-E(\mathbf N).
\]

当前实现已经具有局部增量更新，而不是每一步全量重算。这是后续扩展到大模型的重要基础，但尚未做系统复杂度和性能 benchmark。

### 4.6 模拟退火接受规则

对一次交换：

- 若 \(\Delta E\le0\)，必然接受；
- 若 \(\Delta E>0\)，以

\[
P_{\mathrm{accept}}=\exp(-\Delta E/T)
\]

的概率接受。

初始温度与当前目标值及 `random_temperature` 相乘，随后按

\[
T(s)=T_0\max\left[10^{-4},(1-s/S)^2\right]
\]

降温，其中 (s) 是当前步数，(S) 是该相的总步数。接受少量变差交换是为了跳出局部最优；温度降低后搜索逐渐转为局部精修。

### 4.7 多相和多晶当前实际行为

native SQS 按 `phase_id` 逐相优化，所以支持多相几何；每相内部可以跨不同晶粒交换元素。`region_constraints` 可以按相、晶粒及主区域类型分别设置组成与区域内部 pair 目标。多相模型中的目标组成必须显式指定相，避免把不存在于某一相的元素隐式投影到该相。

当 `exclude_boundaries=true` 时：

- 若内部原子数量足够，晶界/相界位点不参与目标计算；
- 同时也不参与候选交换，因此边界元素实际被冻结；
- 若内部原子太少，程序回退到使用该相的全部原子。

一旦存在区域约束，优化器会在整个相内提出交换，使区域成分变化由相内其他区域补偿，同时保持相的元素总数不变。区域 pair 项可选择 `internal`（两端均在区域内）或 `touching`（至少一端在区域内）；后者适合很薄的晶界层。跨相界化学对仍没有独立的方向性目标。`segregation` 仍是 SQS 之外的确定性后处理功能。有限固定成分目标、自适应温度与停止规则的完整定义见 `NATIVE_SQS_METHOD.md`。

### 4.8 当前执行流程与源码映射

当前一次 native SQS 构建可以用下面的伪代码表示：

```text
build_model
  ├─ 构建固定晶格、晶粒、相和界面几何
  ├─ 按每相目标成分分配精确整数元素数
  ├─ calculate_sro_report -> 保存优化前统一指标
  └─ optimize_native_sqs
       ├─ 对每一个 phase_id 独立处理
       ├─ 选择可交换位点（可排除边界）
       ├─ _shell_edges -> 建立各近邻壳层边
       ├─ _counts_and_targets -> 当前 pair counts 和随机目标
       ├─ 建立 atom -> affected edges 邻接表
       ├─ 重复 phase_steps 次
       │    ├─ 随机选两个不同元素位点 i,j
       │    ├─ 只计算与 i,j 相连边的 Δcounts
       │    ├─ 计算加权 Δobjective
       │    ├─ 按 Metropolis 规则接受或拒绝
       │    └─ 接受时交换元素并提交局部计数变化
       ├─ 写回 ASE chemical symbols
       └─ calculate_sro_report -> 保存优化后统一指标
```

关键源码函数：

| 功能 | 当前函数 |
|---|---|
| 壳层近邻图 | `_shell_edges` |
| 原子对计数与随机目标 | `_counts_and_targets` |
| pair objective | `_objective` |
| Warren–Cowley 报告 | `_warren_cowley` |
| 统一 SRO 报告 | `calculate_sro_report` |
| native 模拟退火 | `optimize_native_sqs` |
| icet/sqsgenerator 入口 | `backends/sqs.py::optimize_with_backend` |

## 5. 当前 native SQS 的优势与限制

### 已实现：多区域受约束联合目标（2026-08-25）

当前实现已从“仅按相整体优化”扩展为区域化联合目标。一次交换仍限定在同一相内，因此严格保持每相元素整数计数；但交换可以跨区域发生，使晶界偏聚目标与晶内补偿成分同时满足。目标函数由三部分组成：

\[
E = E_{\mathrm{phase\ pair}}
  + \sum_r w_r^{\mathrm{pair}}E_{r,\mathrm{pair}}
  + \sum_r w_r^{\mathrm{comp}}E_{r,\mathrm{comp}}.
\]

区域选择器可组合相、晶粒编号以及 `interior`、`grain_boundary`、
`phase_boundary`、`triple_junction`。其中区域pair项使用 `internal` 或 `touching` 边的多壳层有限尺寸随机配对目标；区域成分项使用目标整数期望与实际元素计数的加权平方偏差。Metropolis接受准则使用整个联合目标的增量，因此某个区域的改善不会被错误地当成与全局目标无关的后处理。

目前仍属于pair层级方法：尚未加入三体/四体cluster vector、界面法向分层目标和通用约束求解器。程序会在优化前报告整数舍入下界、元素可用量以及重叠目标的潜在冲突，但不会证明多个重叠约束可同时精确满足。

### 已有优势

- 不依赖外部后端；
- 支持已构建的多晶和多相几何；
- 每相整数成分严格守恒；
- 多壳层二体目标；
- 模拟退火而非纯贪心；
- 固定随机种子可复现；
- 使用局部边增量更新；
- 输出独立 Warren–Cowley、原子对偏差和归一化误差报告。

### 主要限制

- 仅优化二体原子对，不包含三体、四体簇；
- pair 项以目标组成为随机配对目标，尚不能直接输入任意非零 Warren–Cowley 目标；
- 可区分晶内、晶界、相界和三叉晶界，但重叠区域采用主区域优先级，不支持连续距离分层；
- `touching` 可统计跨区域边，但不包含方向性界面目标或独立的跨相化学目标；
- 相内交换可能跨越很远的空间，未加入扩散路径或交换距离约束；
- 固定模型盒和固定晶格，不搜索超胞形状；
- 未给出全局最优性证明；当前停止判据是目标阈值或无改进耐心值；
- 小区域会报告整数舍入误差和元素可用性，但重叠约束仅作保守冲突提示；
- 当前 objective 与 Warren–Cowley 距离不是同一个数学量；
- 已纳入黄金基准性能记录，但尚无并行化和完整消融研究。

## 6. icet 的方法

icet 用 `ClusterSpace` 定义对称性不等价的簇轨道，并用 cluster vector 描述构型。它不局限于 Warren–Cowley pair SRO；根据 cutoffs 可以包含不同阶数的 pair、triplet 等簇相关。

icet 官方 SQS 目标函数采用

\[
Q=-\omega L+
\sum_\alpha|\Gamma_\alpha-\Gamma_\alpha^{\mathrm{target}}|.
\]

第一项奖励从短距离开始连续“精确匹配”的最大 pair 半径，第二项惩罚整个 cluster vector 与目标的绝对偏差。它通过 Monte Carlo trial swaps 和模拟退火搜索，也支持在小规模问题中进行完全枚举。固定超胞时可使用 `generate_sqs_from_supercells`；一般 `generate_sqs` 还可以在多个超胞形状中搜索。

非常重要：`ClusterSpace(structure, cutoffs=[...])` 中的列表项代表不同簇阶数的截止半径，而不是第一、第二近邻壳层列表。例如 `[8.0, 4.0]` 表示 pair 截止和 triplet 截止。AlloyModelForge 曾经误解这一点，现已修正。

### AlloyModelForge 当前 icet 适配器

当前适配器：

- 只允许单相、单晶粒、全周期、identity 取向；
- 用 ASE 构建 FCC/BCC/HCP 原胞；
- 使用一个 pair cutoff 覆盖用户请求的若干近邻壳层；
- 调用 `generate_sqs_from_supercells` 保持 AlloyModelForge 已有盒形；
- 当前故意只匹配二体簇，没有启用 triplet/quadruplet；
- 返回 icet 版本，再用 AlloyModelForge 的统一 SRO 报告独立复核。

因此“icet 本身能做高阶簇”不等于“AlloyModelForge 当前 icet 后端已经做了高阶簇”。

## 7. mcsqs/ATAT 的方法

mcsqs 是 ATAT 中的 SQS 工具。其核心是 Monte Carlo 模拟退火，但方法设计与简单的固定权重误差和不同：它优先寻求从最短程开始精确匹配最大数量/范围的相关函数，同时也控制剩余相关偏差。

其重要能力包括：

- 多组分、多子晶格框架；
- 同时优化位点占据和超胞形状；
- 随机并行搜索；
- 可包含 pair 和多体簇；
- 以精确匹配尽可能多的短程相关为重要目标。

与 AlloyModelForge 当前 native SQS 相比，mcsqs 面向理想周期晶格上的 SQS 超胞搜索，而不是已有任意多晶/多相几何上的区域化优化。后续论文不应宣称 AlloyModelForge 在传统单晶 SQS 上全面优于 mcsqs；更合理的贡献是解决 mcsqs/常规 cluster-space 工具不直接针对的非均匀多区域模型。

## 8. sqsgenerator 的方法

sqsgenerator 0.5.6 直接计算多组分、多壳层 Warren–Cowley SRO。其目标可写为

\[
\mathcal O(\sigma)=
\sum_{i,\xi,\eta}
\tilde p^i_{\xi\eta}
\left|
\tilde\alpha^i_{\xi\eta}-\alpha^i_{\xi\eta}
\right|,
\]

其中可分别指定壳层权重、元素对权重和目标 SRO。它提供两种候选排列模式：随机打乱，以及按字典序系统枚举。官方文档没有把当前随机模式描述为 AlloyModelForge/icet 那样的 Metropolis 模拟退火，因此不能笼统写成“所有工具都采用模拟退火”。

sqsgenerator 的特点包括：

- C++ 核心，强调高速 SRO 计算；
- 随机或系统构型遍历；
- 多线程及可选 MPI；
- 自动或手动识别配位壳层；
- 多组分和多子晶格；
- 可以设置非零 target SRO 和 pair-specific 权重；
- 能与 ASE、pymatgen、pyiron 等互操作。

### AlloyModelForge 当前 sqsgenerator 适配器

当前适配器：

- 只允许单相、单晶粒、全周期、identity 取向；
- 将 AlloyModelForge 现有结构、整数元素计数、迭代次数传入；
- 壳层权重设为 (1/n)；
- 固定单线程和随机种子；
- 尚未暴露 `target_objective`、`pair_weights`、系统枚举、MPI 等高级参数；
- 返回 sqsgenerator 内部 objective，并用 AlloyModelForge 统一指标复核。

所以当前仅使用了 sqsgenerator 能力的一部分。

## 9. 四种方法的直接比较

| 方法 | 搜索对象 | 主要描述量 | 搜索策略 | 超胞形状 | 多晶/多相任意几何 |
|---|---|---|---|---|---|
| AlloyModelForge native | 固定几何上的元素占位 | 全相及多区域多壳层 pair counts/WC-SRO | Metropolis 模拟退火交换 | 固定 | 支持区域组成与区域内部 pair 目标；尚无跨界面 pair 目标 |
| icet | 晶体超胞占位 | 对称性约化 cluster vector | MC trial swaps + 模拟退火；小体系可枚举 | 可搜索或固定 | 标准接口要求结构映射到原胞，不适合一般晶界几何 |
| mcsqs | 晶体超胞形状与占位 | 多体相关函数 | Monte Carlo 模拟退火 | 联合优化 | 主要面向周期晶格 SQS |
| sqsgenerator | 固定晶格位点占位 | 多组分、多壳层 WC-SRO | 随机打乱或系统枚举 | 通常由输入固定 | 可处理一般位点，但当前 AlloyModelForge 适配器限制为规则单晶 |

这个表只比较方法范式，不代表性能排名。真实性能和质量必须用同一组输入、同一独立指标、多随机种子进行 benchmark。

## 10. 可以形成方法创新的核心问题

最值得研究的方向不是再次实现一个普通单晶 SQS，而是：

> 在缺少统一平移对称性的多晶、多相和界面结构中，同时满足全局整数成分、区域成分、区域内部相关和跨区域相关，并用可扩展的局部增量算法优化。

将原子划分为区域 (r\in\mathcal R)：晶粒内部、晶界、相界、三叉晶界、析出相等。可以定义

\[
E=
\sum_{r,n,A,B}w_{r,n,A,B}
\left(\alpha_{AB}^{(r,n)}-alpha_{AB,\mathrm{target}}^{(r,n)}\right)^2
+\sum_{r<s}w_{rs}E_{rs}^{\mathrm{cross}}
+\lambda E_{\mathrm{composition}}.
\]

其中第一项控制区域内部 SRO，第二项控制跨区域原子对，第三项处理区域成分偏差。同时满足

\[
\sum_rN_A^{(r)}=N_A^{\mathrm{global}},
\]

以及区域整数约束

\[
|N_A^{(r)}-c_{A,\mathrm{target}}^{(r)}N^{(r)}|\le\epsilon_r.
\]

### 推荐的创新模块

1. **整数可行性分配器**：先求各区域各元素的整数数量，使全局总数精确守恒，并最小化区域成分舍入误差。
2. **区域近邻图**：每条边保存壳层、两端区域、相、晶粒及边界类型。
3. **区域内部目标**：每个区域具有独立成分和 target SRO。
4. **跨界面目标**：显式控制 FCC–BCC、晶内–晶界、基体–析出相的元素配对。
5. **受约束交换提议**：只产生不会破坏硬性整数约束的交换，或使用成环交换同时满足多个区域约束。
6. **增量 objective**：只更新交换原子邻接的边及相关区域统计。
7. **任意 target SRO**：支持零目标、实验目标、Monte Carlo/cluster-expansion 目标。
8. **收敛与不确定性**：多链、多随机种子、最佳值轨迹、接受率和重复性报告。

## 11. 建议的研究实现顺序

### 阶段 A：把现有算法定义严谨化

- 将优化内部加权 objective 与报告 objective 同时输出；
- 保存每一步或分段的最佳值、当前值、温度、接受率；
- 允许显式设置壳层权重；
- 允许输入任意 pair target SRO；
- 增加目标可实现性和整数成分诊断；
- 建立 32 原子小体系的完全枚举基准，验证模拟退火能否找到全局最优。

### 阶段 B：区域约束

- 为 grain interior、grain boundary、phase boundary、triple junction 建立 region selector；
- 先实现区域成分硬约束；
- 再实现区域内部 pair SRO；
- 最后实现跨区域 pair SRO；
- 测试约束冲突时给出明确的不可行报告，而不是无限搜索。

### 阶段 C：高阶与性能

- 添加局部 triplet 图及增量更新；
- 对比 pair-only 与 pair+triplet 对结构区分能力；
- 多链并行、分区提议或 GPU/Numba/C++ 加速可作为后续性能方向；
- 测试 (10^3,10^4,10^5,10^6) 原子的时间和峰值内存。

## 12. 必须开展的算法验证

### 正确性

- 二元/三元小超胞完全枚举对照；
- 手工可计算构型的 pair counts 与 Warren–Cowley 参数；
- 成分在每次接受交换后严格守恒；
- 增量更新与全量重算逐步一致；
- 周期边界跨盒原子对不漏计、不重复计数；
- 区域内边和跨区域边分类准确。

### 搜索质量

- 至少 10–30 个随机种子；
- 比较随机占位、贪心、模拟退火；
- 比较不同温度计划和步数；
- 与 icet、mcsqs、sqsgenerator 使用统一外部指标比较；
- 报告均值、标准差、最好/最差结果和成功率。

### 消融实验

1. 全局 pair-only；
2. 加区域成分约束；
3. 加区域内部 SRO；
4. 加跨界面 SRO；
5. 加 triplet；
6. 全量重算与增量更新性能对照。

### 科学案例

- CoCrFeMnNi 多晶的晶内/晶界 target SRO；
- FCC/BCC 双相合金的跨相界配对；
- 基体/析出相体系中的界面偏聚；
- 对比结构弛豫前后的 SRO 保持程度。弛豫可在项目外使用 LAMMPS 完成，不要求把 MD 引擎嵌入 AlloyModelForge。

## 13. 需要避免的论文表述

- 不写“把 SRO 降得越低，结构越随机”；应写“最小化与目标 SRO/相关向量的偏差”。
- 不直接比较不同后端的内部 objective 数值。
- 不把随机着色图当成 SQS 证据。
- 不把 `icet` 的全部能力当成当前 AlloyModelForge 适配器能力。
- 不宣称 sqsgenerator 当前随机模式是模拟退火，除非以后依据其具体版本源码重新确认。
- 不把未弛豫多晶界面称为稳定材料微结构；它是用于后续计算的初始几何模型。
- 不声称 pair-only 能唯一确定化学构型。

## 14. 主要资料

1. Zunger, Wei, Ferreira, Bernard, *Special Quasirandom Structures*, Physical Review Letters 65 (1990). [原始论文 PDF](https://www.colorado.edu/faculty/zunger-matter-by-design/sites/default/files/attached-files/170.pdf)
2. van de Walle, Tiwary, de Jong et al., *Efficient stochastic generation of special quasirandom structures*, Calphad 42 (2013), 13–18. [论文页面](https://www.sciencedirect.com/science/article/pii/S0364591613000540)
3. icet, *Special quasirandom structures*. [官方文档](https://icet.materialsmodeling.org/advanced_topics/sqs_generation.html)
4. icet, `structure_generation` API/source documentation. [官方文档](https://icet.materialsmodeling.org/_modules/icet/tools/structure_generation.html)
5. sqsgenerator, *under the hood: Pair Short-Range-Order*. [官方文档](https://sqsgenerator.readthedocs.io/en/latest/under_the_hood.html)
6. sqsgenerator, configuration parameters and search modes. [官方文档](https://sqsgenerator.readthedocs.io/en/latest/parameters.html)

## 15. 当前结论

AlloyModelForge 已经不是最朴素的“每步全量重算并贪心交换”程序：它具有固定成分交换、Metropolis 接受、自适应降温、局部增量更新、有限尺寸随机目标，以及按相、晶粒和界面类型定义的区域组成/`internal`/`touching`/`crossing` pair 联合目标。Native SQS v4 还支持冻结相参考组成的 `source→target` 方向性 Warren–Cowley 目标，以及按近邻壳层和夹角分箱的受限中心角三体。它仍不是 icet 式通用对称 cluster space，尚未包含 quadruplet 和超胞形状搜索。

后续最有价值的研究主线是完善“多区域受约束相关函数优化器”：增加整数可行性预诊断、跨界面 SRO、任意 target SRO 和高阶簇。这一方向应通过小体系严格最优对照、第三方交叉验证、多随机种子统计和规模性能测试来证明，而不能仅以生成成功或可视化效果作为依据。
