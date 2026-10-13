<h1 align="center">strata-batch-boost</h1>

[English](README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · **日本語** · [한국어](README.ko.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b><a href="https://github.com/Niko1221/Strata">Strata</a> の並列リクエストで合計速度が最大 +22%</b><br>
出力はバイト単位で同一 · 単一リクエストの速度は変わらず · 環境変数 2 つ、デフォルトでオフ</p>

[Strata](https://github.com/Niko1221/Strata) は Qwen3.8-Flash-Next を 1 台の PC で動かし、**1 つ**のリクエストを処理する場合は
非常に高速です。複数のリクエストを同時に処理する（`--batch N`）と、各リクエストが本来以上に遅くなります。このパッチは
そのオーバーヘッドの大部分を取り除きます。MTP、QSA attention、VRAM expert キャッシュはまったくそのままです。

| | Strata（upstream） | + このパッチ |
|---|---|---|
| 1 リクエスト | 179 tok/s | 179 tok/s |
| 4 リクエスト同時（`--batch 4`）、合計 | 268 tok/s | **312 tok/s**（+16%） |
| 8 リクエスト同時（`--batch 8`）、合計 | 308 tok/s | **375 tok/s**（+22%） |

Strata main `a0116b2`（2026-10-12）、RTX PRO 6000 Blackwell、Qwen3.8-Flash-Next IQ3_S で測定。Strata
0.1.43 でも同じ傾向です（[結果](#結果)を参照）。

## インストール

```bash
git clone https://github.com/yi-sudo/strata-batch-boost
./strata-batch-boost/apply.sh /path/to/Strata     # your Strata git clone
cd /path/to/Strata && ./setup.sh --build --no-start   # compile the engine from source
```

次に、モデルの `strata-<model>.json` で有効化し（setup がこのファイルを書き換える際もこのブロックは保持されます）、
同時に処理するリクエスト数を設定します。

```json
"parallel": 4,
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

コンパイルには NVIDIA GPU と CUDA toolkit が必要です（パッチは CUDA kernel を変更します）。Strata の `./update.sh` は
`git pull` を行うので、先に Strata で `git checkout -- .` を実行してから更新し、その後もう一度 `apply.sh` を実行してください。

## 並列リクエストが遅かった理由

各 decode ステップで、Strata は実行中の全リクエストの次トークンを 1 つの「verify window」（最大 8 行）にまとめます。

| | window あたりの時間 | window あたりのトークン数 |
|---|---|---|
| 1 リクエスト + MTP | ~6.8 ms | ~1.56（物語）、コードではそれ以上 |
| upstream `--batch`、R 行 | ~4.8 ms + **1 行あたり ~2.4 ms** | R |

1 行あたり 2.4 ms のうち、expert の読み込みは ~0.42 ms にすぎず、これはすでにメモリ帯域の上限です。残りの大部分は
**リクエストごとに順番に実行される小さな GPU launch** です：QSA ブロックスコアリング、top-k と attention、そして
GDN recurrence（window 内とコミット時の両方）。

## パッチの変更内容

| 環境変数 | 内容 |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | QSA ブロックスコア、top-k、decode attention を、それぞれ**全行まとめて 1 回の launch** で実行します。各行は引き続き自分のリクエストの indexer キーと K/V プールを読み、これらは kernel パラメータとして値渡しされるため、行レイアウトごとにキャプチャされた CUDA graph はそのまま有効です。リクエストが KV をストリーミングする場合（`kv_mode 1`）は行ごとの launch にフォールバックします。 |
| `STRATA_BATCH_GDN_GROUPS=1` | window 内の GDN recurrence と 1 行のコミットを、**全リクエストまとめて 1 回の launch** で実行します（`grid.y` = リクエスト）。 |

行ごとの演算は同じなので、temperature 0 での出力はバイト単位で同一です。どちらの変数も設定しなければ、
エンジンは upstream とまったく同じ動作をします。ファイル：`src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`、
そのヘッダー、および `src/core/verify.cpp`（+202 / −6 行）。

## テスト環境

- NVIDIA RTX PRO 6000 Blackwell Workstation（96 GB）、ドライバ 610.57.04、CUDA 13.0、Linux
- Qwen3.8-Flash-Next IQ3_S、24,576 個の expert すべてを VRAM に常駐
- 特記なき限り `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp --vision`、`--vram-reserve-mib 20480`
- 負荷：[`bench_concurrency.py`](lab/bench_concurrency.py) が N 個のストリーミングリクエストを同時に送信します。各リクエストは
  異なる物語プロンプトを使うので、共有される prefix はありません。各リクエストは 1024 トークンを生成し、1 点につき 2 回実行します。

## 結果

### Strata main `a0116b2`（テストした最新版）

合計 tok/s。`--batch 8` の実行では、すべての expert が VRAM に収まるよう 6144 MiB のリザーブを使用。データ：
[latest-a0116b2](lab/results/latest-a0116b2)。

| リクエスト数 | upstream `--batch 4` | パッチ適用 `--batch 4` | upstream `--batch 8` | パッチ適用 `--batch 8` |
|---|---|---|---|---|
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 ¹ | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | **311.7** | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | **374.8** |

¹ 再実行。この点の 1 回目の実行は外乱を受けました（125.1、[NOTE](lab/results/latest-a0116b2/NOTE.md) を参照）。

temperature 0 での出力：`--batch 4`、`--batch 8` とも upstream と `IDENTICAL`。

### Strata 0.1.43（`7695859`）

`--batch 4` での合計 tok/s（[phase4](lab/results/phase4)、[phase5](lab/results/phase5)）：

| リクエスト数 | upstream | attention のみ | attention + GDN |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

`--batch 8` での合計 tok/s（[phase6](lab/results/phase6)）。ここではリザーブを 6144 MiB にしているので、すべての expert が
VRAM に収まります：

| リクエスト数 | upstream | パッチ適用 |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

verify window の時間（ms）、エンジン自身の `strata batch:` 統計より：

| 行数 | upstream | attention のみ | attention + GDN |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6 | 19.25 | | 15.92 |
| 8 | 24.55 | | 19.91 |

### 正確性

[`parity.py`](lab/parity.py) が temperature 0 のプロンプトを 4 つ同時に送信し、[`compare_parity.py`](lab/compare_parity.py)
がパッチ適用版の出力を upstream の出力とバイト単位で比較します。テストしたすべての構成で `IDENTICAL` でした。

### upstream のベースライン：`--batch` と `--batch-mtp`

upstream 0.1.43、20480 MiB リザーブでの合計 tok/s（[production-p1-p4-p8](lab/results/production-p1-p4-p8/summary.txt)）。
`--batch 8` では 24,576 個の expert のうち 17,097 個しか VRAM に収まらないため、1 リクエストの速度が遅くなっています。

| リクエスト数 | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### メモリ：`--batch` は起動時にスロットを確保する

データ：[context-memory](lab/results/context-memory/summary.txt)。

リクエスト実行中に VRAM は増えません。4 つの同時リクエストの前・最中・後で 73,352 MiB のままでした。
各スロットは `--max-context` 全体分の領域を確保するので、`--batch 4` のコストはコンテキストサイズで決まります：

| 設定 | GPU 全体 | 1 リクエスト | 4 リクエスト（512 トークン） |
|---|---|---|---|
| p1, 262K | 59.9 GB | 186 | 170（キュー待ち） |
| p4, 64K | 61.1 GB | 181 | 286 |
| p4, 128K | 65.7 GB | 181 | 291 |
| p4, 262K | ~74.9 GB | 184 | ~309 |

### 効果がなかったアイデア：リクエストあたりの MTP draft を増やす

データ：[phase3a](lab/results/phase3a)、[phase3a2](lab/results/phase3a2)、パッチ：
[`patches/experimental`](patches/experimental)。

このパッチでは、各リクエストが最大 D 個の MTP draft を連鎖させて window に入れられます。batch モードでは draft の
**50–57%** しか受理されませんが、行を 1 つ追加して元が取れるのは ~67% 以上の場合だけです。4 リクエストでの合計速度は
D=1 で 246 tok/s、D=3 で 199、D=5 で 193 で、MTP なしの `--batch 4` の 268 を下回りました。このパッチは参考用としてのみ残しています。

## 再現

[`lab/`](lab) にはこれらの数値を出したすべてのスクリプトがあります。各実行の生 JSON とエンジンログは
[`lab/results/`](lab/results) にあります。

`lab/env.sh`、または git 管理外の `lab/env.local.sh` で `ROOT`（パッチ適用済みの Strata クローン）、`STRATA_DATA`、`PY` を
設定します。`lab-start.sh <parallel>` はポート 8095 でテストサーバーを起動し、`phaseN.sh` は 1 つの実験を実行して
サーバーを再び停止します。

| スクリプト | 測定内容 |
|---|---|
| `phase1.sh` | p1 対 p4 のベースライン（p1 の実行結果 57 tok/s は無効：別のインスタンスがまだ GPU 上にあった） |
| `phase2.sh` | ステージごとの window 時間（`STRATA_DECODE_TIMING`、`STRATA_VERIFY_PROFILE`） |
| `phase3a.sh`, `phase3a2.sh` | 複数 draft の MTP 実験 |
| `phase4.sh` | attention のみ 対 upstream、parity 付き |
| `phase5.sh` | attention + GDN、parity 付き |
| `phase6.sh` | `--batch 8`、upstream 対 パッチ適用 |
| `bench-dashboard.sh`, `context-memory.sh` | ダッシュボード経由で複数の設定で再起動 |

## 制限事項

- テストは 1 つの GPU と 1 つのモデルのみ：単一の RTX PRO 6000、IQ3_S、すべての expert を VRAM に配置。複数 GPU、
  PCIe や CPU 上の expert、AMD ではテストしていません。
- マルチストリーム分岐は意図的に避けています：upstream は Linux オープンモジュールで `STRATA_DF_BRANCH` をオフにしています（#905）。
- conv のコミットと indexer への追記は、まだリクエストごとに 1 回ずつ実行されます。これらをまとめるのが次のステップです。

## ライセンス

MIT。パッチは [Strata](https://github.com/Niko1221/Strata)（MIT、© 2026 Niko1221 and the Strata
contributors）を変更します。[LICENSE](LICENSE) を参照してください。
