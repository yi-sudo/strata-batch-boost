<h1 align="center">strata-batch-boost</h1>

**English** · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Up to +22% total speed for parallel requests in <a href="https://github.com/Niko1221/Strata">Strata</a></b><br>
Same outputs, byte for byte · single-request speed unchanged · two env vars, off by default</p>

[Strata](https://github.com/Niko1221/Strata) runs Qwen3.8-Flash-Next on one PC and is very fast for **one** request
at a time. With several requests at once (`--batch N`), each request slows down more than it should. This patch
removes most of that overhead. MTP, QSA attention and the VRAM expert cache stay exactly as they are.

| | Strata (upstream) | + this patch |
|---|---|---|
| 1 request | 179 tok/s | 179 tok/s |
| 4 requests at once (`--batch 4`), total | 268 tok/s | **312 tok/s** (+16%) |
| 8 requests at once (`--batch 8`), total | 308 tok/s | **375 tok/s** (+22%) |

Measured on Strata main `a0116b2` (2026-10-12) with an RTX PRO 6000 Blackwell and Qwen3.8-Flash-Next IQ3_S. Strata
0.1.43 gives the same picture (see [Results](#results)).

## Install

```bash
git clone https://github.com/yi-sudo/strata-batch-boost
./strata-batch-boost/apply.sh /path/to/Strata     # your Strata git clone
cd /path/to/Strata && ./setup.sh --build --no-start   # compile the engine from source
```

Then turn it on in your model's `strata-<model>.json` (setup keeps this block when it rewrites the file) and set how
many requests run at once:

```json
"parallel": 4,
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

Needs an NVIDIA GPU and a CUDA toolkit to compile (the patch changes CUDA kernels). Strata's `./update.sh` does a
`git pull`: run `git checkout -- .` in Strata first, update, then run `apply.sh` again.

## Why parallel requests were slow

Each decode step, Strata puts the next token of every running request into one "verify window" (up to 8 rows):

| | time per window | tokens per window |
|---|---|---|
| 1 request + MTP | ~6.8 ms | ~1.56 (stories), more for code |
| upstream `--batch`, R rows | ~4.8 ms + **~2.4 ms per row** | R |

Of the 2.4 ms per row, reading the experts takes only ~0.42 ms, which is already the memory bandwidth limit. Most of
the rest is **small GPU launches that run one request after another**: QSA block scoring, top-k and attention, and the
GDN recurrence, both in the window and when it commits.

## What the patch changes

| env var | what it does |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | QSA block scores, top-k and decode attention for **all rows in one launch each**. Each row still reads its own request's indexer keys and K/V pools, which are passed by value as kernel params, so the CUDA graph captured per row layout stays valid. Falls back to per-row launches when a request streams its KV (`kv_mode 1`). |
| `STRATA_BATCH_GDN_GROUPS=1` | GDN recurrence in the window, and the 1-row commit, for **all requests in one launch** (`grid.y` = request). |

The per-row arithmetic is the same, so temperature-0 outputs are byte-identical. With neither variable set, the
engine behaves exactly like upstream. Files: `src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`,
their headers, and `src/core/verify.cpp` (+202 / −6 lines).

## Test setup

- NVIDIA RTX PRO 6000 Blackwell Workstation (96 GB), driver 610.57.04, CUDA 13.0, Linux
- Qwen3.8-Flash-Next IQ3_S, all 24,576 experts resident in VRAM
- `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp --vision`, `--vram-reserve-mib 20480` unless noted
- Load: [`bench_concurrency.py`](lab/bench_concurrency.py) sends N streaming requests at the same time. Each request
  gets a different story prompt, so no prefix is shared. Each produces 1024 tokens, with 2 runs per point.

## Results

### Strata main `a0116b2` (newest tested)

Total tok/s. The `--batch 8` runs use a 6144 MiB reserve so that all experts still fit in VRAM. Data:
[latest-a0116b2](lab/results/latest-a0116b2).

| requests | upstream `--batch 4` | patched `--batch 4` | upstream `--batch 8` | patched `--batch 8` |
|---|---|---|---|---|
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 ¹ | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | **311.7** | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | **374.8** |

¹ Rerun. The first run of this point was disturbed (125.1, see [NOTE](lab/results/latest-a0116b2/NOTE.md)).

Outputs at temperature 0: `IDENTICAL` to upstream for both `--batch 4` and `--batch 8`.

### Strata 0.1.43 (`7695859`)

Total tok/s with `--batch 4` ([phase4](lab/results/phase4), [phase5](lab/results/phase5)):

| requests | upstream | attention only | attention + GDN |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

Total tok/s with `--batch 8` ([phase6](lab/results/phase6)). The reserve is 6144 MiB here, so all experts still fit
in VRAM:

| requests | upstream | patched |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

Verify window time in ms, from the engine's own `strata batch:` stats:

| rows | upstream | attention only | attention + GDN |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6 | 19.25 | | 15.92 |
| 8 | 24.55 | | 19.91 |

### Correctness

[`parity.py`](lab/parity.py) sends 4 concurrent temperature-0 prompts. [`compare_parity.py`](lab/compare_parity.py)
then compares the patched outputs with upstream's, byte by byte. Every configuration tested gave `IDENTICAL`.

### Upstream baseline: `--batch` and `--batch-mtp`

Total tok/s on upstream 0.1.43 with a 20480 MiB reserve ([production-p1-p4-p8](lab/results/production-p1-p4-p8/summary.txt)).
With `--batch 8` only 17,097 of the 24,576 experts fit in VRAM, which is why one request is slower there.

| requests | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### Memory: `--batch` reserves its slots at startup

Data: [context-memory](lab/results/context-memory/summary.txt).

VRAM does not grow while requests run. It stayed at 73,352 MiB before, during and after 4 concurrent requests.
Each slot reserves room for the full `--max-context`, so the context size decides what `--batch 4` costs:

| setting | whole GPU | 1 request | 4 requests (512 tokens) |
|---|---|---|---|
| p1, 262K | 59.9 GB | 186 | 170 (queued) |
| p4, 64K | 61.1 GB | 181 | 286 |
| p4, 128K | 65.7 GB | 181 | 291 |
| p4, 262K | ~74.9 GB | 184 | ~309 |

### An idea that did not pay: more MTP drafts per request

Data: [phase3a](lab/results/phase3a), [phase3a2](lab/results/phase3a2), patch:
[`patches/experimental`](patches/experimental).

This patch lets each request chain up to D MTP drafts into the window. In batch mode only **50–57%** of the drafts
are accepted, but an extra row only pays off above ~67%. With 4 requests the total speed was 246 tok/s at D=1,
199 at D=3 and 193 at D=5, against 268 for `--batch 4` without MTP. The patch is kept for reference only.

## Reproduce

[`lab/`](lab) has every script behind these numbers. Each run's raw JSON and engine log are in
[`lab/results/`](lab/results).

Set `ROOT` (the patched Strata clone), `STRATA_DATA` and `PY` in `lab/env.sh`, or in an untracked
`lab/env.local.sh`. `lab-start.sh <parallel>` starts a test server on port 8095, and `phaseN.sh` runs one experiment
and stops the server again.

| script | measures |
|---|---|
| `phase1.sh` | p1 vs p4 baseline (its p1 run, 57 tok/s, is invalid: another instance was still on the GPU) |
| `phase2.sh` | per-stage window time (`STRATA_DECODE_TIMING`, `STRATA_VERIFY_PROFILE`) |
| `phase3a.sh`, `phase3a2.sh` | the multi-draft MTP experiment |
| `phase4.sh` | attention only vs upstream, with parity |
| `phase5.sh` | attention + GDN, with parity |
| `phase6.sh` | `--batch 8`, upstream vs patched |
| `bench-dashboard.sh`, `context-memory.sh` | restart through a dashboard at several settings |

## Limits

- Tested on one GPU and one model: a single RTX PRO 6000 with IQ3_S and every expert in VRAM. It has not been tested
  with multiple GPUs, with experts on PCIe or the CPU, or on AMD.
- Multi-stream branches were avoided on purpose: upstream turns `STRATA_DF_BRANCH` off on Linux open modules (#905).
- The conv commits and the indexer appends still run once per request. Grouping them is the next step.

## License

MIT. The patches modify [Strata](https://github.com/Niko1221/Strata) (MIT, © 2026 Niko1221 and the Strata
contributors). See [LICENSE](LICENSE).
