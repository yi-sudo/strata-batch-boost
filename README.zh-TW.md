<h1 align="center">strata-batch-boost</h1>

[English](README.md) · **繁體中文** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>讓 <a href="https://github.com/Niko1221/Strata">Strata</a> 的平行請求總速度最多提升 22%</b><br>
輸出逐位元組完全相同 · 單一請求速度不變 · 兩個環境變數，預設關閉</p>

[Strata](https://github.com/Niko1221/Strata) 能在一台 PC 上執行 Qwen3.8-Flash-Next，**一次一個**請求時速度非常快。
但同時處理多個請求（`--batch N`）時，每個請求變慢的幅度超出應有的程度。這個 patch
消除了大部分這類額外開銷。MTP、QSA attention 與 VRAM expert 快取都完全維持原樣。

| | Strata（上游） | + 本 patch |
|---|---|---|
| 1 個請求 | 179 tok/s | 179 tok/s |
| 同時 4 個請求（`--batch 4`），總計 | 268 tok/s | **312 tok/s**（+16%） |
| 同時 8 個請求（`--batch 8`），總計 | 308 tok/s | **375 tok/s**（+22%） |

測量環境為 Strata main `a0116b2`（2026-10-12），搭配 RTX PRO 6000 Blackwell 與 Qwen3.8-Flash-Next IQ3_S。Strata
0.1.43 的結果趨勢相同（見[測試結果](#測試結果)）。

## 安裝

```bash
git clone https://github.com/yi-sudo/strata-batch-boost
./strata-batch-boost/apply.sh /path/to/Strata     # your Strata git clone
cd /path/to/Strata && ./setup.sh --build --no-start   # compile the engine from source
```

接著在模型的 `strata-<model>.json` 中啟用它（setup 重寫該檔案時會保留這個區塊），並設定
同時執行的請求數：

```json
"parallel": 4,
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

編譯需要 NVIDIA 顯示卡與 CUDA toolkit（本 patch 修改了 CUDA kernel）。Strata 的 `./update.sh` 會執行
`git pull`：請先在 Strata 中執行 `git checkout -- .`，更新後再重新執行 `apply.sh`。

## 為什麼平行請求會變慢

每個 decode 步驟中，Strata 會把每個執行中請求的下一個 token 放進同一個「verify window」（最多 8 列）：

| | 每個 window 耗時 | 每個 window 的 token 數 |
|---|---|---|
| 1 個請求 + MTP | ~6.8 ms | ~1.56（故事），程式碼更多 |
| 上游 `--batch`，R 列 | ~4.8 ms + **每列 ~2.4 ms** | R |

在每列 2.4 ms 中，讀取 expert 只佔約 0.42 ms，這已經是記憶體頻寬的極限。其餘大部分是
**逐個請求依序執行的小型 GPU launch**：QSA block 評分、top-k 與 attention，以及
GDN 遞迴，在 window 內與 commit 時都有。

## 本 patch 修改了什麼

| 環境變數 | 作用 |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | QSA block 評分、top-k 與 decode attention 各自**以一次 launch 處理所有列**。每列仍讀取自己請求的 indexer key 與 K/V pool，這些以傳值方式作為 kernel 參數傳入，因此依列配置擷取的 CUDA graph 仍然有效。當請求以串流方式讀取 KV（`kv_mode 1`）時，退回逐列 launch。 |
| `STRATA_BATCH_GDN_GROUPS=1` | window 內的 GDN 遞迴與 1 列 commit，**以一次 launch 處理所有請求**（`grid.y` = 請求）。 |

每列的運算完全相同，因此 temperature 0 的輸出逐位元組一致。兩個變數都未設定時，
引擎行為與上游完全相同。檔案：`src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`、
對應的標頭檔，以及 `src/core/verify.cpp`（+202 / −6 行）。

## 測試環境

- NVIDIA RTX PRO 6000 Blackwell Workstation（96 GB），驅動程式 610.57.04，CUDA 13.0，Linux
- Qwen3.8-Flash-Next IQ3_S，全部 24,576 個 expert 常駐 VRAM
- `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp --vision`，除非另有註明，皆使用 `--vram-reserve-mib 20480`
- 負載：[`bench_concurrency.py`](lab/bench_concurrency.py) 同時送出 N 個串流請求。每個請求
  使用不同的故事提示詞，因此沒有共用前綴。每個請求產生 1024 個 token，每個資料點跑 2 次。

## 測試結果

### Strata main `a0116b2`（測試過的最新版）

總 tok/s。`--batch 8` 的測試使用 6144 MiB 保留量，讓所有 expert 仍能放進 VRAM。資料：
[latest-a0116b2](lab/results/latest-a0116b2)。

| 請求數 | 上游 `--batch 4` | 加 patch `--batch 4` | 上游 `--batch 8` | 加 patch `--batch 8` |
|---|---|---|---|---|
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 ¹ | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | **311.7** | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | **374.8** |

¹ 重跑。這個資料點的第一次執行受到干擾（125.1，見 [NOTE](lab/results/latest-a0116b2/NOTE.md)）。

temperature 0 的輸出：`--batch 4` 與 `--batch 8` 皆與上游 `IDENTICAL`。

### Strata 0.1.43（`7695859`）

`--batch 4` 的總 tok/s（[phase4](lab/results/phase4)、[phase5](lab/results/phase5)）：

| 請求數 | 上游 | 僅 attention | attention + GDN |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

`--batch 8` 的總 tok/s（[phase6](lab/results/phase6)）。這裡保留量為 6144 MiB，讓所有 expert 仍能放進
VRAM：

| 請求數 | 上游 | 加 patch |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

verify window 耗時（ms），取自引擎本身的 `strata batch:` 統計：

| 列數 | 上游 | 僅 attention | attention + GDN |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6 | 19.25 | | 15.92 |
| 8 | 24.55 | | 19.91 |

### 正確性

[`parity.py`](lab/parity.py) 同時送出 4 個 temperature 0 的提示詞。接著 [`compare_parity.py`](lab/compare_parity.py)
逐位元組比較加 patch 後與上游的輸出。所有測試過的設定皆為 `IDENTICAL`。

### 上游基準：`--batch` 與 `--batch-mtp`

上游 0.1.43 在 20480 MiB 保留量下的總 tok/s（[production-p1-p4-p8](lab/results/production-p1-p4-p8/summary.txt)）。
使用 `--batch 8` 時，24,576 個 expert 中只有 17,097 個能放進 VRAM，這就是該設定下單一請求較慢的原因。

| 請求數 | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### 記憶體：`--batch` 在啟動時就預留 slot

資料：[context-memory](lab/results/context-memory/summary.txt)。

請求執行期間 VRAM 不會增長。在 4 個並行請求之前、期間與之後都維持在 73,352 MiB。
每個 slot 都預留完整 `--max-context` 的空間，因此 context 大小決定了 `--batch 4` 的成本：

| 設定 | 整張顯示卡 | 1 個請求 | 4 個請求（512 token） |
|---|---|---|---|
| p1, 262K | 59.9 GB | 186 | 170（排隊） |
| p4, 64K | 61.1 GB | 181 | 286 |
| p4, 128K | 65.7 GB | 181 | 291 |
| p4, 262K | ~74.9 GB | 184 | ~309 |

### 沒有成效的嘗試：每個請求更多 MTP 草稿

資料：[phase3a](lab/results/phase3a)、[phase3a2](lab/results/phase3a2)，patch：
[`patches/experimental`](patches/experimental)。

這個 patch 讓每個請求能在 window 中串接最多 D 個 MTP 草稿。在 batch 模式下只有 **50–57%** 的草稿
被接受，但多一列要在接受率約 67% 以上才划算。4 個請求時，總速度在 D=1 為 246 tok/s，
D=3 為 199，D=5 為 193，而不開 MTP 的 `--batch 4` 為 268。此 patch 僅保留作為參考。

## 重現

[`lab/`](lab) 包含產生這些數據的所有腳本。每次執行的原始 JSON 與引擎日誌都在
[`lab/results/`](lab/results)。

在 `lab/env.sh` 或未納入版本控制的 `lab/env.local.sh` 中設定 `ROOT`（加 patch 的 Strata clone）、`STRATA_DATA` 與 `PY`。
`lab-start.sh <parallel>` 會在 port 8095 啟動測試伺服器，`phaseN.sh` 執行一項實驗
後再停止伺服器。

| 腳本 | 測量內容 |
|---|---|
| `phase1.sh` | p1 vs p4 基準（其中 p1 的結果 57 tok/s 無效：當時另一個實例仍佔用顯示卡） |
| `phase2.sh` | 各階段的 window 耗時（`STRATA_DECODE_TIMING`、`STRATA_VERIFY_PROFILE`） |
| `phase3a.sh`、`phase3a2.sh` | 多草稿 MTP 實驗 |
| `phase4.sh` | 僅 attention vs 上游，含一致性檢查 |
| `phase5.sh` | attention + GDN，含一致性檢查 |
| `phase6.sh` | `--batch 8`，上游 vs 加 patch |
| `bench-dashboard.sh`、`context-memory.sh` | 透過 dashboard 以多種設定重新啟動 |

## 限制

- 只在一張顯示卡與一個模型上測試：單張 RTX PRO 6000，IQ3_S，所有 expert 都在 VRAM。尚未在
  多顯示卡、expert 放在 PCIe 或 CPU、或 AMD 上測試。
- 刻意避開多 stream 分支：上游在 Linux open module 上關閉了 `STRATA_DF_BRANCH`（#905）。
- conv commit 與 indexer append 仍是每個請求各執行一次。將它們合併是下一步。

## 授權

MIT。這些 patch 修改自 [Strata](https://github.com/Niko1221/Strata)（MIT，© 2026 Niko1221 and the Strata
contributors）。見 [LICENSE](LICENSE)。
