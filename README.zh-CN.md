<h1 align="center">strata-batch-boost</h1>

[English](README.md) · [繁體中文](README.zh-TW.md) · **简体中文** · [日本語](README.ja.md) · [한국어](README.ko.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>让 <a href="https://github.com/Niko1221/Strata">Strata</a> 的并行请求总速度最多提升 22%</b><br>
输出逐字节完全相同 · 单个请求速度不变 · 两个环境变量，默认关闭</p>

[Strata](https://github.com/Niko1221/Strata) 可以在一台 PC 上运行 Qwen3.8-Flash-Next，**一次一个**请求时速度非常快。
但同时处理多个请求（`--batch N`）时，每个请求变慢的幅度超出了应有的程度。这个 patch
消除了大部分此类额外开销。MTP、QSA attention 和 VRAM expert 缓存都完全保持原样。

| | Strata（上游） | + 本 patch |
|---|---|---|
| 1 个请求 | 179 tok/s | 179 tok/s |
| 同时 4 个请求（`--batch 4`），总计 | 268 tok/s | **312 tok/s**（+16%） |
| 同时 8 个请求（`--batch 8`），总计 | 308 tok/s | **375 tok/s**（+22%） |

测量环境为 Strata main `a0116b2`（2026-10-12），搭配 RTX PRO 6000 Blackwell 和 Qwen3.8-Flash-Next IQ3_S。Strata
0.1.43 的结果趋势相同（见[测试结果](#测试结果)）。

## 安装

```bash
git clone https://github.com/yi-sudo/strata-batch-boost
./strata-batch-boost/apply.sh /path/to/Strata     # your Strata git clone
cd /path/to/Strata && ./setup.sh --build --no-start   # compile the engine from source
```

然后在模型的 `strata-<model>.json` 中启用它（setup 重写该文件时会保留这个块），并设置
同时运行的请求数：

```json
"parallel": 4,
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

编译需要 NVIDIA 显卡和 CUDA toolkit（本 patch 修改了 CUDA kernel）。Strata 的 `./update.sh` 会执行
`git pull`：请先在 Strata 中运行 `git checkout -- .`，更新后再重新运行 `apply.sh`。

## 为什么并行请求会变慢

每个 decode 步骤中，Strata 会把每个运行中请求的下一个 token 放进同一个“verify window”（最多 8 行）：

| | 每个 window 耗时 | 每个 window 的 token 数 |
|---|---|---|
| 1 个请求 + MTP | ~6.8 ms | ~1.56（故事），代码更多 |
| 上游 `--batch`，R 行 | ~4.8 ms + **每行 ~2.4 ms** | R |

在每行 2.4 ms 中，读取 expert 只占约 0.42 ms，这已经是显存带宽的极限。其余大部分是
**逐个请求依次执行的小型 GPU launch**：QSA block 打分、top-k 和 attention，以及
GDN 递推，在 window 内和 commit 时都有。

## 本 patch 改了什么

| 环境变量 | 作用 |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | QSA block 打分、top-k 和 decode attention 各自**用一次 launch 处理所有行**。每行仍读取自己请求的 indexer key 和 K/V pool，它们按值作为 kernel 参数传入，因此按行布局捕获的 CUDA graph 依然有效。当请求以流式方式读取 KV（`kv_mode 1`）时，回退为逐行 launch。 |
| `STRATA_BATCH_GDN_GROUPS=1` | window 内的 GDN 递推和 1 行 commit，**用一次 launch 处理所有请求**（`grid.y` = 请求）。 |

每行的计算完全相同，因此 temperature 0 的输出逐字节一致。两个变量都未设置时，
引擎行为与上游完全相同。文件：`src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`、
对应的头文件，以及 `src/core/verify.cpp`（+202 / −6 行）。

## 测试环境

- NVIDIA RTX PRO 6000 Blackwell Workstation（96 GB），驱动 610.57.04，CUDA 13.0，Linux
- Qwen3.8-Flash-Next IQ3_S，全部 24,576 个 expert 常驻 VRAM
- `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp --vision`，除非另有说明，均使用 `--vram-reserve-mib 20480`
- 负载：[`bench_concurrency.py`](lab/bench_concurrency.py) 同时发送 N 个流式请求。每个请求
  使用不同的故事提示词，因此没有共享前缀。每个请求生成 1024 个 token，每个数据点跑 2 次。

## 测试结果

### Strata main `a0116b2`（测试过的最新版）

总 tok/s。`--batch 8` 的测试使用 6144 MiB 预留量，使所有 expert 仍能放进 VRAM。数据：
[latest-a0116b2](lab/results/latest-a0116b2)。

| 请求数 | 上游 `--batch 4` | 打 patch `--batch 4` | 上游 `--batch 8` | 打 patch `--batch 8` |
|---|---|---|---|---|
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 ¹ | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | **311.7** | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | **374.8** |

¹ 重跑。该数据点的第一次运行受到干扰（125.1，见 [NOTE](lab/results/latest-a0116b2/NOTE.md)）。

temperature 0 的输出：`--batch 4` 和 `--batch 8` 均与上游 `IDENTICAL`。

### Strata 0.1.43（`7695859`）

`--batch 4` 的总 tok/s（[phase4](lab/results/phase4)、[phase5](lab/results/phase5)）：

| 请求数 | 上游 | 仅 attention | attention + GDN |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

`--batch 8` 的总 tok/s（[phase6](lab/results/phase6)）。这里预留量为 6144 MiB，使所有 expert 仍能放进
VRAM：

| 请求数 | 上游 | 打 patch |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

verify window 耗时（ms），取自引擎自身的 `strata batch:` 统计：

| 行数 | 上游 | 仅 attention | attention + GDN |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6 | 19.25 | | 15.92 |
| 8 | 24.55 | | 19.91 |

### 正确性

[`parity.py`](lab/parity.py) 同时发送 4 个 temperature 0 的提示词。然后 [`compare_parity.py`](lab/compare_parity.py)
逐字节比较打 patch 后与上游的输出。所有测试过的配置均为 `IDENTICAL`。

### 上游基线：`--batch` 和 `--batch-mtp`

上游 0.1.43 在 20480 MiB 预留量下的总 tok/s（[production-p1-p4-p8](lab/results/production-p1-p4-p8/summary.txt)）。
使用 `--batch 8` 时，24,576 个 expert 中只有 17,097 个能放进 VRAM，这就是该配置下单个请求较慢的原因。

| 请求数 | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### 显存：`--batch` 在启动时就预留 slot

数据：[context-memory](lab/results/context-memory/summary.txt)。

请求运行期间 VRAM 不会增长。在 4 个并发请求之前、期间和之后都保持在 73,352 MiB。
每个 slot 都预留完整 `--max-context` 的空间，因此 context 大小决定了 `--batch 4` 的开销：

| 设置 | 整张显卡 | 1 个请求 | 4 个请求（512 token） |
|---|---|---|---|
| p1, 262K | 59.9 GB | 186 | 170（排队） |
| p4, 64K | 61.1 GB | 181 | 286 |
| p4, 128K | 65.7 GB | 181 | 291 |
| p4, 262K | ~74.9 GB | 184 | ~309 |

### 没有收益的尝试：每个请求更多 MTP 草稿

数据：[phase3a](lab/results/phase3a)、[phase3a2](lab/results/phase3a2)，patch：
[`patches/experimental`](patches/experimental)。

这个 patch 让每个请求可以在 window 中串接最多 D 个 MTP 草稿。在 batch 模式下只有 **50–57%** 的草稿
被接受，但多一行要在接受率约 67% 以上才划算。4 个请求时，总速度在 D=1 为 246 tok/s，
D=3 为 199，D=5 为 193，而不开 MTP 的 `--batch 4` 为 268。该 patch 仅保留作参考。

## 复现

[`lab/`](lab) 包含产生这些数据的所有脚本。每次运行的原始 JSON 和引擎日志都在
[`lab/results/`](lab/results)。

在 `lab/env.sh` 或未纳入版本控制的 `lab/env.local.sh` 中设置 `ROOT`（打 patch 的 Strata clone）、`STRATA_DATA` 和 `PY`。
`lab-start.sh <parallel>` 会在端口 8095 启动测试服务器，`phaseN.sh` 运行一项实验
后再停止服务器。

| 脚本 | 测量内容 |
|---|---|
| `phase1.sh` | p1 vs p4 基线（其中 p1 的结果 57 tok/s 无效：当时另一个实例仍占用显卡） |
| `phase2.sh` | 各阶段的 window 耗时（`STRATA_DECODE_TIMING`、`STRATA_VERIFY_PROFILE`） |
| `phase3a.sh`、`phase3a2.sh` | 多草稿 MTP 实验 |
| `phase4.sh` | 仅 attention vs 上游，含一致性检查 |
| `phase5.sh` | attention + GDN，含一致性检查 |
| `phase6.sh` | `--batch 8`，上游 vs 打 patch |
| `bench-dashboard.sh`、`context-memory.sh` | 通过 dashboard 以多种配置重启 |

## 局限

- 只在一张显卡和一个模型上测试：单张 RTX PRO 6000，IQ3_S，所有 expert 都在 VRAM。尚未在
  多显卡、expert 放在 PCIe 或 CPU、或 AMD 上测试。
- 刻意避开了多 stream 分支：上游在 Linux open module 上关闭了 `STRATA_DF_BRANCH`（#905）。
- conv commit 和 indexer append 仍是每个请求各执行一次。将它们合并是下一步。

## 许可证

MIT。这些 patch 修改自 [Strata](https://github.com/Niko1221/Strata)（MIT，© 2026 Niko1221 and the Strata
contributors）。见 [LICENSE](LICENSE)。
