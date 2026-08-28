# AlloyModelForge

**面向 MD、DFT 与高通量工作流的合金原子计算模型构建与验证平台**  
**High-throughput construction and validation of atomistic alloy models for downstream MD and DFT**

[![Tests](https://img.shields.io/badge/tests-81%20passed-brightgreen)](#测试与科学验证)
[![Version](https://img.shields.io/badge/version-1.0-blueviolet)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](environment.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Interface](https://img.shields.io/badge/interface-Web%20%7C%20CLI-orange)](#两种使用方式)

AlloyModelForge 是一个本地优先（local-first）的开源科学软件，用于构建可供后续分子动力学（MD）、密度泛函理论（DFT）或其他原子模拟使用的初始合金结构。软件将单晶、多晶、多相、析出相、核–壳颗粒、缺陷、化学无序和短程有序（SRO/SQS）放在统一、可验证、可复现的模型规范中。

软件同时提供：

- 面向新手的中文/英文五步 Web 向导；
- 面向批处理、复现和高性能计算流程的 CLI；
- YAML 工程文件和 Python API；
- 几何、组成、SRO、CNA/PTM、RDF、文件往返及性能验证；
- 适合论文图像的 OVITO 分析和渲染接口。

> [!IMPORTANT]
> AlloyModelForge 研究和验证的是**初始原子模型的构建**。势函数选择、能量最小化、退火、相稳定性、动力学演化和电子结构计算不属于本软件的研究范围。多晶界面和异相界面通常仍需在后续 MD/DFT 工作流中弛豫。

## 目录

- [主要能力](#主要能力)
- [方法定位](#方法定位)
- [快速开始](#快速开始)
- [两种使用方式](#两种使用方式)
- [示例模型](#示例模型)
- [YAML 配置说明](#yaml-配置说明)
- [输出文件](#输出文件)
- [测试与科学验证](#测试与科学验证)
- [在另一台电脑上复现](#在另一台电脑上复现)
- [上传 GitHub](#上传-github)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [引用、贡献与许可证](#引用贡献与许可证)

## 主要能力

### 几何与微结构

- FCC、BCC、HCP 单晶；
- 任意正交长方体计算盒，不局限于正方体；
- 周期性 Voronoi 多晶与可复现随机取向；
- FCC/BCC 等按晶粒分配的多相多晶；
- 球、椭球、长方体、圆柱和层状析出区域；
- 同中心嵌套区域构建核–壳颗粒；
- 空位、指定位置间隙原子和三叉区标记；
- CIF、POSCAR 或其他 ASE 可读结构作为自定义晶胞模板。

### 化学无序与 SQS

- 严格整数元素数目的随机占位；
- AlloyModelForge Native SQS；
- 多近邻壳层 pair SRO/Warren–Cowley 优化；
- 晶粒、晶内、晶界、相界和三叉区的区域组成/SRO 约束；
- 任意方向性 SRO 约束；
- triplet 目标；
- 自适应退火温度、多次独立启动和提前停止；
- MILP 联合区域组成可行性检查；
- 可选 icet 和 sqsgenerator 后端，且不会静默替换算法。

### 验证与可视化

- 原子数、元素组成、相分数和区域分布；
- 最近邻距离、配位数、数密度、局部密度和 RDF；
- 周期接缝、短近邻、界面比例和晶粒统计；
- 核–壳径向相分数与结构有效性指标；
- OVITO CNA/PTM 分析；
- 按元素、晶粒、相、区域、界面、CNA 或 PTM 着色；
- OVITO Tachyon 高分辨率渲染和 OVITO Desktop 打开功能；
- EXTXYZ、LAMMPS data、POSCAR 和 CIF 往返读取检查；
- 随机种子、软件版本、依赖版本和结构 SHA-256 指纹记录。

## 方法定位

AlloyModelForge 不是 ASE、OVITO、icet 或 sqsgenerator 的替代品，而是面向复杂初始合金模型的统一构建和验证层：

| 组件 | 在 AlloyModelForge 中的作用 |
|---|---|
| ASE | 原子结构数据模型、晶体操作和格式读写 |
| OVITO | 独立 CNA/PTM 分析及期刊级渲染 |
| icet | 规则晶格 cluster-space SQS 可选后端 |
| sqsgenerator | 多组元规则晶格 SQS 可选后端 |
| Native SQS | 面向真实几何邻居、区域约束、多晶界面和核壳区域的内置优化方法 |

软件的主要定位是：通过统一 YAML 规范构建复杂几何，在构建阶段提供明确的科学验收指标，并让 Web、CLI、Python API 和论文验证脚本使用同一模型定义。

## 快速开始

### 1. 前置条件

推荐配置：

- Windows 10/11 或现代 Linux x86-64；
- Miniconda、Anaconda 或 Miniforge；
- Git；
- 至少 8 GB 内存（大型模型建议 16 GB 或更多）；
- 支持 WebGL 的现代浏览器；
- 磁盘预留至少 2 GB 用于完整环境和测试输出。

项目的权威运行环境是 Conda 环境 `alloymodelforge-dev`。不要安装到 Conda `base`，也不需要创建 `.venv`。

### 2. 克隆项目

```powershell
git clone https://github.com/caojian1134/AlloyModelForge.git
cd AlloyModelForge
```

### 3. 创建完整科学环境

Windows PowerShell：

```powershell
conda env create -f environment.yml
conda activate alloymodelforge-dev
python -m pip check
pytest -q
```

如果环境已经存在：

```powershell
conda env update -n alloymodelforge-dev -f environment.yml --prune
conda activate alloymodelforge-dev
python -m pip check
pytest -q
```

也可以在 Windows 中使用自动脚本。它会创建/更新环境，并执行依赖、测试和复现检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap-dev.ps1
```

Linux 终端使用：

```bash
conda env create -f environment.yml
conda activate alloymodelforge-dev
python -m pip check
pytest -q
```

> [!NOTE]
> `environment.yml` 固定了经过测试的 Python、icet 和 OVITO 版本。只运行基础构建时可使用 `pip install -e ".[web,test]"`，但论文复现和跨电脑比较应使用 Conda 完整环境。
> 无图形/Vulkan 后端的服务器或 CI 环境可使用 `pytest -q -m "not ovito_render"`；真实 OVITO Tachyon 渲染仍应在具备图形后端的完整环境中单独验收。

## 两种使用方式

### Web：适合新手与交互建模

Windows：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-web.ps1
```

或者在已激活环境中运行：

```powershell
conda activate alloymodelforge-dev
alloy-model-forge-web --host 127.0.0.1 --port 8765 --no-browser
```

浏览器打开 `http://127.0.0.1:8765`。

Web 默认显示中文，右上角 `EN / 中文` 按钮可即时切换语言。推荐的新手流程为：

1. 从顶部载入一个结构模板；
2. 选择单晶、多晶或双相模型；
3. 设置长方体尺寸、周期方向、晶粒数量和随机种子；
4. 编辑晶体结构、晶格参数、元素组成和析出/核壳区域；
5. 选择 Random、Native SQS、icet 或 sqsgenerator；
6. 设置几何清理、CNA/PTM、RDF 和输出格式；
7. 构建模型并检查三维预览、验证指标和输出文件。

“完整高级配置”区域允许直接生成、编辑并重新应用 YAML。Web 表单没有直接展示的区域 SQS、方向性 SRO、triplet 和缺陷字段会被保留，因此 Web 与 CLI 不会产生不兼容的配置版本。

Web 任务默认保存在 `%USERPROFILE%\.alloymodelforge\jobs`。旧版本的
`%USERPROFILE%\.alloyforge\jobs` 不会被自动删除。可临时指定其他目录：

```powershell
$env:ALLOYMODELFORGE_DATA_DIR="D:\alloy-model-forge-jobs"
alloy-model-forge-web --host 127.0.0.1 --port 8765 --no-browser
```

默认只监听 `127.0.0.1`，不会向公网提供服务。如果需要局域网访问，应自行配置主机防火墙和可信网络，不建议将当前本地服务直接暴露到互联网。

### CLI：适合复现、批处理和自动化

```powershell
# 估算模型规模
alloy-model-forge estimate examples\fcc_polycrystal.yaml

# 构建模型
alloy-model-forge build examples\fcc_polycrystal.yaml --output output\polycrystal

# 独立验证已有模型
alloy-model-forge validate output\polycrystal\model.extxyz output\polycrystal\specification.yaml

# 重新分析已有模型
alloy-model-forge analyze output\polycrystal\model.extxyz output\polycrystal\specification.yaml

# 快速查看、OVITO 打开和渲染
alloy-model-forge view output\polycrystal\model.extxyz
alloy-model-forge open-ovito output\polycrystal\model.extxyz
alloy-model-forge render-ovito output\polycrystal\model.extxyz --output figure.png

# 查看工程元数据
alloy-model-forge project-info output\polycrystal

# 从 CLI 启动 Web
alloy-model-forge web --host 127.0.0.1 --port 8765
```

### Python API

```python
from alloymodelforge import build_model, load_specification

specification = load_specification("examples/fcc_rectangular_cell.yaml")
result = build_model(specification)
result.write_outputs("output/rectangular-cell")
```

正式 Python 导入名是 `alloymodelforge`。为保证旧工程与早期脚本继续运行，
`from alloyforge import ...`、`alloyforge` 和 `alloyforge-web` 仍作为兼容别名保留；
新项目应优先使用 `alloymodelforge`、`alloy-model-forge` 和
`alloy-model-forge-web`。

## 示例模型

| 文件 | 内容 | 适合用途 |
|---|---|---|
| `fcc_rectangular_cell.yaml` | 14.4 × 18.0 × 21.6 Å³ 标准 FCC 长方体 | 安装和格式检查 |
| `hcp_ti_single_crystal.yaml` | HCP Ti 单晶 | HCP 几何与 c/a 检查 |
| `fcc_polycrystal.yaml` | FCC 多晶 | 晶粒、晶界和随机取向 |
| `fcc_bcc_dual_phase.yaml` | FCC/BCC 双相多晶 | 相分配与相界验证 |
| `fcc_bcc_spherical_precipitate.yaml` | FCC 基体中的 BCC 球形析出相 | 析出区域验证 |
| `fcc_bcc_hcp_core_shell.yaml` | FCC 基体/BCC 壳/HCP 核 | 核–壳径向验证 |
| `fcc_cocrfeni_native_sqs.yaml` | CoCrFeNi FCC Native SQS | SRO 优化基准 |
| `fcc_polycrystal_region_sqs.yaml` | 多晶区域约束 SQS | 晶内/晶界联合约束 |
| `fcc_bcc_hcp_core_shell_directional_sqs.yaml` | 核壳、方向性和 triplet 联合案例 | 高级方法验证 |

例如构建标准长方体：

```powershell
alloy-model-forge build examples\fcc_rectangular_cell.yaml --output output\rectangular
```

当前参考结果应包含 480 个原子，验证状态为 `PASS`，最近邻距离约为 2.546 Å。

## YAML 配置说明

最小配置示例：

```yaml
name: CoCrFeNi FCC example
random_seed: 19
box:
  lengths: [14.36, 14.36, 14.36]
  periodic: [true, true, true]
phases:
  - name: FCC
    structure: fcc
    lattice_parameter: 3.59
    fraction: 1.0
    composition: {Co: 0.25, Cr: 0.25, Fe: 0.25, Ni: 0.25}
grains:
  number: 1
  orientations: identity
chemistry:
  method: native_sqs
  neighbor_shells: 2
  steps: 10000
  restarts: 1
analysis:
  ovito_enabled: true
  methods: [cna, ptm]
  rdf_enabled: true
output:
  formats: [extxyz, lammps-data, vasp, cif]
```

关键原则：

- `box.lengths` 为三个正数，单位是 Å；
- 所有相的 `fraction` 之和必须为 1；
- 每个相的 `composition` 之和必须为 1；
- 相名必须唯一，析出区域通过相名引用目标相；
- 使用同一输入、软件版本和随机种子可复现模型；
- 对 SQS 方法，应保存优化前后目标值，而不应只根据元素着色图判断质量。

完整字段和高级约束可参考 [`examples/`](examples) 以及 [`docs/SRO_ALGORITHM_RESEARCH.md`](docs/SRO_ALGORITHM_RESEARCH.md)。

## 输出文件

| 文件 | 内容 |
|---|---|
| `model.extxyz` | 保留元素、晶粒、相、区域和结构标签的主结构文件 |
| `model.data` | LAMMPS data 文件 |
| `POSCAR` | VASP 结构文件 |
| `model.cif` | CIF 结构文件 |
| `specification.yaml` | 标准化后的完整输入规范 |
| `specification.json` | JSON 形式的完整输入规范 |
| `metadata.json` | 软件版本、结构指纹、SQS、OVITO 和验证元数据 |
| `validation.json` | 独立结构有效性报告 |
| `optimization-history.csv` | SQS 优化轨迹（启用时） |
| `output-roundtrip.json` | 各输出格式重新读取后的完整性检查 |

验证状态含义：

- `PASS`：所有预声明构建指标通过；
- `WARNING`：模型成功生成且没有结构错误，但存在需要解释的科学限制；
- `FAIL`：存在不可接受的组成、几何、输出或结构问题。

小尺寸随机取向多晶常因正交周期盒不完全共格、界面占比高或少量短近邻得到 `WARNING`。不要为了得到 `PASS` 而随意放宽阈值；论文模型应增大尺寸，并完整报告警告和后续弛豫要求。

## 测试与科学验证

### 快速回归测试

```powershell
conda run -n alloymodelforge-dev pytest -q
```

当前版本在完整 Conda 环境中为 **81 passed**。第三方库可能产生 deprecation warning；只要测试返回码为 0，就不等同于模型验证失败。

### 固定结构复现检查

```powershell
conda run -n alloymodelforge-dev python scripts\reproducibility_check.py
```

该检查比较固定输入的原子数、组成、最小距离、验证状态和 SHA-256 结构指纹，可用于判断另一台电脑是否得到相同结果。

### Web 真实端到端测试

先在一个终端启动 Web：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-web.ps1
```

再在另一个终端运行：

```powershell
conda run -n alloymodelforge-dev python scripts\validate_web_workflows.py
```

该脚本不是只测试 HTTP 路由，而是通过 Web API 逐个提交八类真实结构，等待独立计算完成，并检查必需输出、原子预览、报告一致性和结构错误。报告写入 `validation-output/web-e2e/report.json`。当前参考运行结果为 8/8 工作流完成，共构建 16,648 个原子。

### 科学与期刊基准

```powershell
conda run -n alloymodelforge-dev python scripts\scientific_validation.py --output scientific-validation.json
conda run -n alloymodelforge-dev python scripts\ovito_gold_standard.py
conda run -n alloymodelforge-dev python scripts\publication_benchmark.py
conda run -n alloymodelforge-dev python scripts\journal_validation.py
```

这些脚本分别覆盖理想 FCC/BCC/HCP、SQS、CNA/PTM、复杂结构、文件往返、图像和性能数据。详细发表边界见 [`docs/PUBLICATION_WORKFLOW.md`](docs/PUBLICATION_WORKFLOW.md)。

### GitHub Actions

仓库包含 [`.github/workflows/tests.yml`](.github/workflows/tests.yml)。每次 push 和 pull request 会在 Windows 与 Ubuntu 创建 Conda 环境，执行无图形依赖的自动化测试、科学验证和期刊验证，并保存验证产物。真实 OVITO Tachyon 渲染测试需要兼容的 Vulkan/图形后端，因此不在 GitHub 托管运行器中执行，而保留在本地图形环境的完整验收链中。发布前应在 GitHub 仓库的 **Actions** 页面确认两个操作系统均通过。

## 在另一台电脑上复现

建议严格按照以下验收顺序测试，而不是复制本机已有环境：

1. 在新电脑安装 Git 和 Miniconda/Miniforge；
2. 从 GitHub 重新 `git clone`，不要复制 `.venv` 或 Conda 环境；
3. 使用 `environment.yml` 创建全新 `alloymodelforge-dev`；
4. 运行 `python -m pip check` 和 `pytest -q`；
5. 运行 `scripts/reproducibility_check.py`；
6. CLI 构建标准长方体并验证；
7. 启动 Web，切换中英文并构建同一模板；
8. 如果资源允许，再运行 Web 八模板和期刊基准。

完整验收命令：

```powershell
git clone https://github.com/caojian1134/AlloyModelForge.git
cd AlloyModelForge
conda env create -f environment.yml
conda activate alloymodelforge-dev
python -m pip check
pytest -q
python scripts\reproducibility_check.py
alloy-model-forge build examples\fcc_rectangular_cell.yaml --output acceptance-output\rectangular
alloy-model-forge validate acceptance-output\rectangular\model.extxyz acceptance-output\rectangular\specification.yaml
```

新电脑验收时建议记录操作系统、CPU、依赖版本、测试数量、复现指纹、长方体原子数、八模板成功数量、耗时、峰值内存及所有 `WARNING`。如果不同平台的浮点计算导致结构字节指纹不同，应进一步比较原子数、整数组成、几何容差、SRO 和结构统计，不要只依据 PNG 判断一致性。

## 上传 GitHub

### 1. 上传前排除生成内容

不要上传 `.venv/`、Conda 环境、Web 任务、`validation-output/`、测试输出、大型图片、缓存、日志、私人路径、令牌或未公开数据。这些目录已写入 `.gitignore`。首次提交前检查：

```powershell
git status --short
git ls-files | Select-String -Pattern "\.venv|validation-output|test-output|conversation-recovery"
```

第二条命令正常情况下不应返回内容。

### 2. 提交版本更新

```powershell
git add <需要发布的文件>
git status
git commit -m "Release AlloyModelForge 1.0"
git push origin main
```

提交前务必阅读 `git status`，确认没有环境、测试输出或隐私文件被加入。

### 3. 核对远程仓库

项目远程仓库为 <https://github.com/caojian1134/AlloyModelForge>。推送前核对远程地址与当前分支：

```powershell
git remote -v
git branch --show-current
git status --short
```

仓库当前保持私有；只有在作者明确决定公开并完成许可证、作者信息、数据与安全检查后，才应修改可见性。

### 4. 1.0 发布检查

- 核对 `pyproject.toml`、Python API、Web API 和 `CITATION.cff` 的版本均为 `1.0`；
- 核对发布日期、项目 URL、作者信息和 DOI（如有）；
- 确认 MIT `LICENSE` 符合发布意图；
- 确认本地图形环境中的完整测试和真实 OVITO 渲染通过；
- 确认 GitHub Actions 在 Windows 和 Ubuntu 均通过；
- 创建 `v1.0` 标签与 GitHub Release；
- 如需可引用 DOI，可连接 Zenodo 后创建正式 Release。

```powershell
git tag -a v1.0 -m "AlloyModelForge 1.0"
git push origin v1.0
```

## 常见问题

### Conda 环境创建较慢或依赖冲突

确认使用 `environment.yml`，不要混用 base、`.venv` 和项目环境。必要时可删除失败的 Conda 环境后重建：

```powershell
conda env remove -n alloymodelforge-dev
conda env create -f environment.yml
```

该命令会删除 Conda 环境，执行前确认环境名正确。

### Windows CLI 帮助文字出现乱码

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
alloy-model-forge --help
```

项目提供的 PowerShell 脚本已经设置这两个变量。

### Web 页面打不开

确认终端仍在运行，并检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/health
```

端口被占用时使用 `alloy-model-forge-web --host 127.0.0.1 --port 8766 --no-browser`。

### 3D 查看器没有显示

先确认任务已完成并存在 `model.extxyz`，然后检查浏览器 WebGL、网络资源和控制台。即使交互组件不可用，结构文件、验证报告和 OVITO 离线渲染仍可独立使用。

### OVITO 分析不可用

```powershell
conda run -n alloymodelforge-dev python -c "import ovito; print(ovito.version_string)"
```

不要在同一环境混用 PyPI OVITO wheel 和 Conda OVITO，否则 Windows 可能出现 Qt/DLL 冲突。

### 为什么模型是 WARNING？

`WARNING` 通常表示模型可输出，但存在需要判断的条件，例如随机取向与正交周期盒不完全共格、模型太小、界面占比高或未弛豫界面有少量短近邻。建议增大模型、检查 OVITO 图像和数值指标，并在后续模拟前弛豫；不要只为消除警告而放宽科学阈值。

## 项目结构

```text
alloy-model-forge/
├── src/alloyforge/          # 核心模型、构建器、SQS、验证和 CLI
├── src/alloymodelforge/     # 正式 Python API 名称及兼容导出
├── src/alloyforge_web/      # FastAPI 服务与中英文 Web 界面
├── examples/                # 可直接运行的 YAML 示例
├── tests/                   # 单元、集成和科学不变量测试
├── scripts/                 # 复现、Web E2E 和期刊验证脚本
├── benchmarks/              # 基准定义与结果规范
├── docs/                    # 方法、发表工作流和研究说明
├── .github/workflows/       # Windows/Linux 持续集成
├── environment.yml          # 权威 Conda 科学环境
├── pyproject.toml           # Python 打包与命令入口
├── CITATION.cff             # GitHub 引用元数据
├── CONTRIBUTING.md          # 贡献规范
└── LICENSE                  # MIT License
```

## 引用、贡献与许可证

如果 AlloyModelForge 对研究工作有帮助，请引用 [`CITATION.cff`](CITATION.cff)。论文正式发表或归档到 Zenodo 后，应将真实作者、版本、URL 和 DOI 写入该文件。

欢迎提交 issue 和 pull request。科学方法改动必须同时提供物理不变量、独立参考结果或可解释的数值基准；仅检查函数不报错或文件存在不足以证明模型科学有效。详细要求见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

AlloyModelForge 使用 [MIT License](LICENSE)。第三方依赖及其许可证由各自项目维护。

## 延伸文档

- [科学环境与复现指纹](ENVIRONMENT.md)
- [SRO 与 Native SQS 方法研究](docs/SRO_ALGORITHM_RESEARCH.md)
- [论文工作流与验收边界](docs/PUBLICATION_WORKFLOW.md)
- [版本变更记录](CHANGELOG.md)
