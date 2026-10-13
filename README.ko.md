<h1 align="center">strata-batch-boost</h1>

[English](README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **한국어** · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b><a href="https://github.com/Niko1221/Strata">Strata</a>에서 병렬 요청의 전체 속도 최대 +22%</b><br>
출력은 바이트 단위로 동일 · 단일 요청 속도는 그대로 · 환경 변수 2개, 기본값은 꺼짐</p>

[Strata](https://github.com/Niko1221/Strata)는 PC 한 대에서 Qwen3.8-Flash-Next를 실행하며, 한 번에 **하나의** 요청을
처리할 때 매우 빠릅니다. 여러 요청을 동시에 처리하면(`--batch N`) 각 요청이 필요 이상으로 느려집니다. 이 패치는
그 오버헤드의 대부분을 제거합니다. MTP, QSA attention, VRAM expert 캐시는 전혀 바뀌지 않습니다.

| | Strata (upstream) | + 이 패치 |
|---|---|---|
| 요청 1개 | 179 tok/s | 179 tok/s |
| 동시 요청 4개 (`--batch 4`), 합계 | 268 tok/s | **312 tok/s** (+16%) |
| 동시 요청 8개 (`--batch 8`), 합계 | 308 tok/s | **375 tok/s** (+22%) |

Strata main `a0116b2`(2026-10-12), RTX PRO 6000 Blackwell, Qwen3.8-Flash-Next IQ3_S로 측정했습니다. Strata
0.1.43에서도 같은 양상입니다([결과](#결과) 참고).

## 설치

```bash
git clone https://github.com/yi-sudo/strata-batch-boost
./strata-batch-boost/apply.sh /path/to/Strata     # your Strata git clone
cd /path/to/Strata && ./setup.sh --build --no-start   # compile the engine from source
```

그런 다음 모델의 `strata-<model>.json`에서 활성화하고(setup이 이 파일을 다시 쓸 때도 이 블록은 유지됩니다)
동시에 실행할 요청 수를 설정합니다.

```json
"parallel": 4,
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

컴파일하려면 NVIDIA GPU와 CUDA toolkit이 필요합니다(패치가 CUDA kernel을 변경합니다). Strata의 `./update.sh`는
`git pull`을 수행하므로, 먼저 Strata에서 `git checkout -- .`을 실행한 뒤 업데이트하고, 그다음 `apply.sh`를 다시 실행하세요.

## 병렬 요청이 느렸던 이유

각 decode 단계에서 Strata는 실행 중인 모든 요청의 다음 토큰을 하나의 "verify window"(최대 8행)에 넣습니다.

| | window당 시간 | window당 토큰 수 |
|---|---|---|
| 요청 1개 + MTP | ~6.8 ms | ~1.56 (스토리), 코드는 더 많음 |
| upstream `--batch`, R행 | ~4.8 ms + **행당 ~2.4 ms** | R |

행당 2.4 ms 중 expert를 읽는 데는 ~0.42 ms밖에 걸리지 않으며, 이는 이미 메모리 대역폭 한계입니다. 나머지의 대부분은
**요청마다 하나씩 차례로 실행되는 작은 GPU launch**입니다: QSA 블록 스코어링, top-k와 attention, 그리고
GDN recurrence(window 안에서와 커밋할 때 모두).

## 패치가 바꾸는 것

| 환경 변수 | 하는 일 |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | QSA 블록 점수, top-k, decode attention을 각각 **모든 행에 대해 한 번의 launch**로 실행합니다. 각 행은 여전히 자기 요청의 indexer 키와 K/V 풀을 읽으며, 이들은 kernel 파라미터로 값 전달되므로 행 레이아웃별로 캡처된 CUDA graph가 그대로 유효합니다. 요청이 KV를 스트리밍할 때(`kv_mode 1`)는 행별 launch로 폴백합니다. |
| `STRATA_BATCH_GDN_GROUPS=1` | window 안의 GDN recurrence와 1행 커밋을 **모든 요청에 대해 한 번의 launch**로 실행합니다(`grid.y` = 요청). |

행별 연산은 동일하므로 temperature 0 출력은 바이트 단위로 동일합니다. 두 변수 모두 설정하지 않으면
엔진은 upstream과 정확히 똑같이 동작합니다. 파일: `src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`,
해당 헤더, 그리고 `src/core/verify.cpp`(+202 / −6줄).

## 테스트 환경

- NVIDIA RTX PRO 6000 Blackwell Workstation (96 GB), 드라이버 610.57.04, CUDA 13.0, Linux
- Qwen3.8-Flash-Next IQ3_S, 24,576개 expert 전부 VRAM에 상주
- 별도 표기가 없으면 `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp --vision`, `--vram-reserve-mib 20480`
- 부하: [`bench_concurrency.py`](lab/bench_concurrency.py)가 스트리밍 요청 N개를 동시에 보냅니다. 각 요청은
  서로 다른 스토리 프롬프트를 받으므로 공유되는 prefix가 없습니다. 각 요청은 1024 토큰을 생성하며, 측정점마다 2회 실행합니다.

## 결과

### Strata main `a0116b2` (테스트한 최신 버전)

전체 tok/s. `--batch 8` 실행은 모든 expert가 여전히 VRAM에 들어가도록 6144 MiB 예약을 사용합니다. 데이터:
[latest-a0116b2](lab/results/latest-a0116b2).

| 요청 수 | upstream `--batch 4` | 패치 `--batch 4` | upstream `--batch 8` | 패치 `--batch 8` |
|---|---|---|---|---|
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 ¹ | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | **311.7** | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | **374.8** |

¹ 재실행. 이 측정점의 첫 번째 실행은 방해를 받았습니다(125.1, [NOTE](lab/results/latest-a0116b2/NOTE.md) 참고).

temperature 0 출력: `--batch 4`와 `--batch 8` 모두 upstream과 `IDENTICAL`.

### Strata 0.1.43 (`7695859`)

`--batch 4`에서의 전체 tok/s([phase4](lab/results/phase4), [phase5](lab/results/phase5)):

| 요청 수 | upstream | attention만 | attention + GDN |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

`--batch 8`에서의 전체 tok/s([phase6](lab/results/phase6)). 여기서는 예약이 6144 MiB이므로 모든 expert가 여전히
VRAM에 들어갑니다:

| 요청 수 | upstream | 패치 |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

verify window 시간(ms), 엔진 자체의 `strata batch:` 통계 기준:

| 행 수 | upstream | attention만 | attention + GDN |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6 | 19.25 | | 15.92 |
| 8 | 24.55 | | 19.91 |

### 정확성

[`parity.py`](lab/parity.py)가 temperature 0 프롬프트 4개를 동시에 보냅니다. 그다음 [`compare_parity.py`](lab/compare_parity.py)가
패치된 출력을 upstream 출력과 바이트 단위로 비교합니다. 테스트한 모든 구성에서 `IDENTICAL`이었습니다.

### upstream 기준선: `--batch`와 `--batch-mtp`

upstream 0.1.43, 20480 MiB 예약에서의 전체 tok/s([production-p1-p4-p8](lab/results/production-p1-p4-p8/summary.txt)).
`--batch 8`에서는 24,576개 expert 중 17,097개만 VRAM에 들어가므로 요청 1개일 때 더 느립니다.

| 요청 수 | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### 메모리: `--batch`는 시작 시 슬롯을 예약한다

데이터: [context-memory](lab/results/context-memory/summary.txt).

요청이 실행되는 동안 VRAM은 늘어나지 않습니다. 동시 요청 4개의 전, 도중, 후 모두 73,352 MiB로 유지되었습니다.
각 슬롯은 `--max-context` 전체만큼의 공간을 예약하므로, `--batch 4`의 비용은 컨텍스트 크기가 결정합니다:

| 설정 | GPU 전체 | 요청 1개 | 요청 4개 (512 토큰) |
|---|---|---|---|
| p1, 262K | 59.9 GB | 186 | 170 (대기열) |
| p4, 64K | 61.1 GB | 181 | 286 |
| p4, 128K | 65.7 GB | 181 | 291 |
| p4, 262K | ~74.9 GB | 184 | ~309 |

### 효과가 없었던 아이디어: 요청당 MTP draft 늘리기

데이터: [phase3a](lab/results/phase3a), [phase3a2](lab/results/phase3a2), 패치:
[`patches/experimental`](patches/experimental).

이 패치는 각 요청이 최대 D개의 MTP draft를 연결해 window에 넣을 수 있게 합니다. batch 모드에서는 draft의 **50–57%**만
수락되지만, 행을 하나 추가해서 이득을 보려면 ~67% 이상이어야 합니다. 요청 4개에서 전체 속도는 D=1에서 246 tok/s,
D=3에서 199, D=5에서 193으로, MTP 없는 `--batch 4`의 268보다 낮았습니다. 이 패치는 참고용으로만 남겨 둡니다.

## 재현

[`lab/`](lab)에 이 수치를 만든 모든 스크립트가 있습니다. 각 실행의 원본 JSON과 엔진 로그는
[`lab/results/`](lab/results)에 있습니다.

`lab/env.sh` 또는 git에서 추적하지 않는 `lab/env.local.sh`에서 `ROOT`(패치된 Strata 클론), `STRATA_DATA`, `PY`를
설정합니다. `lab-start.sh <parallel>`은 포트 8095에서 테스트 서버를 시작하고, `phaseN.sh`는 실험 하나를 실행한 뒤
서버를 다시 중지합니다.

| 스크립트 | 측정 내용 |
|---|---|
| `phase1.sh` | p1 대 p4 기준선(p1 실행 결과 57 tok/s는 무효: 다른 인스턴스가 아직 GPU에 있었음) |
| `phase2.sh` | 단계별 window 시간(`STRATA_DECODE_TIMING`, `STRATA_VERIFY_PROFILE`) |
| `phase3a.sh`, `phase3a2.sh` | 다중 draft MTP 실험 |
| `phase4.sh` | attention만 대 upstream, parity 포함 |
| `phase5.sh` | attention + GDN, parity 포함 |
| `phase6.sh` | `--batch 8`, upstream 대 패치 |
| `bench-dashboard.sh`, `context-memory.sh` | 대시보드를 통해 여러 설정으로 재시작 |

## 제한 사항

- GPU 하나와 모델 하나에서만 테스트했습니다: 단일 RTX PRO 6000, IQ3_S, 모든 expert를 VRAM에 배치. 다중 GPU,
  PCIe나 CPU에 있는 expert, AMD에서는 테스트하지 않았습니다.
- 멀티 스트림 분기는 의도적으로 피했습니다: upstream은 Linux 오픈 모듈에서 `STRATA_DF_BRANCH`를 끕니다(#905).
- conv 커밋과 indexer 추가는 여전히 요청마다 한 번씩 실행됩니다. 이를 묶는 것이 다음 단계입니다.

## 라이선스

MIT. 패치는 [Strata](https://github.com/Niko1221/Strata)(MIT, © 2026 Niko1221 and the Strata
contributors)를 수정합니다. [LICENSE](LICENSE)를 참고하세요.
