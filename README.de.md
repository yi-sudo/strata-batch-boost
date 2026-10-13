<h1 align="center">strata-batch-boost</h1>

[English](README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · **Deutsch** · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Bis zu +22 % Gesamtgeschwindigkeit bei parallelen Anfragen in <a href="https://github.com/Niko1221/Strata">Strata</a></b><br>
Byte für Byte identische Ausgaben · Geschwindigkeit bei einer einzelnen Anfrage unverändert · zwei Umgebungsvariablen, standardmäßig aus</p>

[Strata](https://github.com/Niko1221/Strata) führt Qwen3.8-Flash-Next auf einem einzelnen PC aus und ist bei **einer**
Anfrage gleichzeitig sehr schnell. Bei mehreren gleichzeitigen Anfragen (`--batch N`) wird jede Anfrage stärker
ausgebremst als nötig. Dieser Patch entfernt den Großteil dieses Overheads. MTP, QSA-Attention und der VRAM-Experten-Cache
bleiben genau so, wie sie sind.

| | Strata (upstream) | + dieser Patch |
|---|---|---|
| 1 Anfrage | 179 tok/s | 179 tok/s |
| 4 Anfragen gleichzeitig (`--batch 4`), gesamt | 268 tok/s | **312 tok/s** (+16 %) |
| 8 Anfragen gleichzeitig (`--batch 8`), gesamt | 308 tok/s | **375 tok/s** (+22 %) |

Gemessen auf Strata main `a0116b2` (2026-10-12) mit einer RTX PRO 6000 Blackwell und Qwen3.8-Flash-Next IQ3_S. Strata
0.1.43 zeigt dasselbe Bild (siehe [Ergebnisse](#ergebnisse)).

## Installation

```bash
git clone https://github.com/yi-sudo/strata-batch-boost
./strata-batch-boost/apply.sh /path/to/Strata     # your Strata git clone
cd /path/to/Strata && ./setup.sh --build --no-start   # compile the engine from source
```

Danach in der `strata-<model>.json` deines Modells einschalten (Setup behält diesen Block bei, wenn es die Datei neu
schreibt) und festlegen, wie viele Anfragen gleichzeitig laufen:

```json
"parallel": 4,
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

Zum Kompilieren werden eine NVIDIA-GPU und ein CUDA-Toolkit benötigt (der Patch ändert CUDA-Kernel). Stratas
`./update.sh` führt ein `git pull` aus: zuerst in Strata `git checkout -- .` ausführen, aktualisieren und dann
`apply.sh` erneut ausführen.

## Warum parallele Anfragen langsam waren

In jedem Decode-Schritt legt Strata das nächste Token jeder laufenden Anfrage in ein gemeinsames „Verify-Fenster“
(bis zu 8 Zeilen):

| | Zeit pro Fenster | Tokens pro Fenster |
|---|---|---|
| 1 Anfrage + MTP | ~6,8 ms | ~1,56 (Geschichten), mehr bei Code |
| upstream `--batch`, R Zeilen | ~4,8 ms + **~2,4 ms pro Zeile** | R |

Von den 2,4 ms pro Zeile entfallen nur ~0,42 ms auf das Lesen der Experten, und das liegt bereits an der Grenze der
Speicherbandbreite. Der Großteil des Rests sind **kleine GPU-Launches, die eine Anfrage nach der anderen abarbeiten**:
QSA-Block-Scoring, top-k und Attention sowie die GDN-Rekurrenz, sowohl im Fenster als auch beim Commit.

## Was der Patch ändert

| Umgebungsvariable | Wirkung |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | QSA-Block-Scores, top-k und Decode-Attention für **alle Zeilen mit je einem Launch**. Jede Zeile liest weiterhin die Indexer-Keys und K/V-Pools ihrer eigenen Anfrage, die als Kernel-Parameter per Wert übergeben werden, sodass der pro Zeilenlayout aufgezeichnete CUDA graph gültig bleibt. Fällt auf Launches pro Zeile zurück, wenn eine Anfrage ihren KV streamt (`kv_mode 1`). |
| `STRATA_BATCH_GDN_GROUPS=1` | GDN-Rekurrenz im Fenster sowie der 1-Zeilen-Commit für **alle Anfragen in einem Launch** (`grid.y` = Anfrage). |

Die Arithmetik pro Zeile bleibt gleich, daher sind die Ausgaben bei Temperatur 0 byte-identisch. Ist keine der beiden
Variablen gesetzt, verhält sich die Engine exakt wie upstream. Dateien:
`src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`, die zugehörigen Header und `src/core/verify.cpp`
(+202 / −6 Zeilen).

## Testumgebung

- NVIDIA RTX PRO 6000 Blackwell Workstation (96 GB), Treiber 610.57.04, CUDA 13.0, Linux
- Qwen3.8-Flash-Next IQ3_S, alle 24.576 Experten resident im VRAM
- `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp --vision`, `--vram-reserve-mib 20480`, sofern nicht anders angegeben
- Last: [`bench_concurrency.py`](lab/bench_concurrency.py) sendet N Streaming-Anfragen gleichzeitig. Jede Anfrage
  erhält einen anderen Geschichten-Prompt, sodass kein Präfix geteilt wird. Jede erzeugt 1024 Tokens, mit 2 Läufen pro Messpunkt.

## Ergebnisse

### Strata main `a0116b2` (neueste getestete Version)

Gesamt-tok/s. Die Läufe mit `--batch 8` nutzen eine Reserve von 6144 MiB, damit alle Experten weiterhin in den VRAM
passen. Daten: [latest-a0116b2](lab/results/latest-a0116b2).

| Anfragen | upstream `--batch 4` | gepatcht `--batch 4` | upstream `--batch 8` | gepatcht `--batch 8` |
|---|---|---|---|---|
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 ¹ | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | **311.7** | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | **374.8** |

¹ Wiederholungslauf. Der erste Lauf dieses Messpunkts war gestört (125.1, siehe [NOTE](lab/results/latest-a0116b2/NOTE.md)).

Ausgaben bei Temperatur 0: `IDENTICAL` zu upstream, sowohl für `--batch 4` als auch für `--batch 8`.

### Strata 0.1.43 (`7695859`)

Gesamt-tok/s mit `--batch 4` ([phase4](lab/results/phase4), [phase5](lab/results/phase5)):

| Anfragen | upstream | nur Attention | Attention + GDN |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

Gesamt-tok/s mit `--batch 8` ([phase6](lab/results/phase6)). Die Reserve beträgt hier 6144 MiB, damit alle Experten
weiterhin in den VRAM passen:

| Anfragen | upstream | gepatcht |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

Zeit des Verify-Fensters in ms, aus den engineeigenen `strata batch:`-Statistiken:

| Zeilen | upstream | nur Attention | Attention + GDN |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6 | 19.25 | | 15.92 |
| 8 | 24.55 | | 19.91 |

### Korrektheit

[`parity.py`](lab/parity.py) sendet 4 gleichzeitige Prompts mit Temperatur 0. [`compare_parity.py`](lab/compare_parity.py)
vergleicht anschließend die gepatchten Ausgaben Byte für Byte mit denen von upstream. Jede getestete Konfiguration
ergab `IDENTICAL`.

### Upstream-Baseline: `--batch` und `--batch-mtp`

Gesamt-tok/s auf upstream 0.1.43 mit einer Reserve von 20480 MiB ([production-p1-p4-p8](lab/results/production-p1-p4-p8/summary.txt)).
Mit `--batch 8` passen nur 17.097 der 24.576 Experten in den VRAM, weshalb eine einzelne Anfrage dort langsamer ist.

| Anfragen | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### Speicher: `--batch` reserviert seine Slots beim Start

Daten: [context-memory](lab/results/context-memory/summary.txt).

Der VRAM-Verbrauch wächst nicht, während Anfragen laufen. Er blieb vor, während und nach 4 gleichzeitigen Anfragen bei
73.352 MiB. Jeder Slot reserviert Platz für den vollen `--max-context`, daher bestimmt die Kontextgröße, was `--batch 4`
kostet:

| Einstellung | gesamte GPU | 1 Anfrage | 4 Anfragen (512 Tokens) |
|---|---|---|---|
| p1, 262K | 59.9 GB | 186 | 170 (in Warteschlange) |
| p4, 64K | 61.1 GB | 181 | 286 |
| p4, 128K | 65.7 GB | 181 | 291 |
| p4, 262K | ~74.9 GB | 184 | ~309 |

### Eine Idee, die sich nicht gelohnt hat: mehr MTP-Drafts pro Anfrage

Daten: [phase3a](lab/results/phase3a), [phase3a2](lab/results/phase3a2), Patch:
[`patches/experimental`](patches/experimental).

Dieser Patch erlaubt jeder Anfrage, bis zu D MTP-Drafts im Fenster zu verketten. Im Batch-Modus werden nur **50–57 %**
der Drafts akzeptiert, eine zusätzliche Zeile lohnt sich aber erst ab ~67 %. Mit 4 Anfragen lag die Gesamtgeschwindigkeit
bei 246 tok/s mit D=1, 199 mit D=3 und 193 mit D=5, gegenüber 268 für `--batch 4` ohne MTP. Der Patch wird nur als
Referenz aufbewahrt.

## Reproduzieren

[`lab/`](lab) enthält alle Skripte hinter diesen Zahlen. Das rohe JSON und das Engine-Log jedes Laufs liegen in
[`lab/results/`](lab/results).

`ROOT` (der gepatchte Strata-Clone), `STRATA_DATA` und `PY` in `lab/env.sh` oder in einer nicht versionierten
`lab/env.local.sh` setzen. `lab-start.sh <parallel>` startet einen Testserver auf Port 8095, und `phaseN.sh` führt ein
Experiment aus und stoppt den Server danach wieder.

| Skript | misst |
|---|---|
| `phase1.sh` | p1 vs. p4 Baseline (der p1-Lauf mit 57 tok/s ist ungültig: eine andere Instanz lief noch auf der GPU) |
| `phase2.sh` | Fensterzeit pro Stufe (`STRATA_DECODE_TIMING`, `STRATA_VERIFY_PROFILE`) |
| `phase3a.sh`, `phase3a2.sh` | das Multi-Draft-MTP-Experiment |
| `phase4.sh` | nur Attention vs. upstream, mit Parität |
| `phase5.sh` | Attention + GDN, mit Parität |
| `phase6.sh` | `--batch 8`, upstream vs. gepatcht |
| `bench-dashboard.sh`, `context-memory.sh` | Neustart über ein Dashboard mit verschiedenen Einstellungen |

## Einschränkungen

- Getestet auf einer GPU mit einem Modell: eine einzelne RTX PRO 6000 mit IQ3_S und allen Experten im VRAM. Nicht
  getestet mit mehreren GPUs, mit Experten über PCIe oder auf der CPU oder auf AMD.
- Multi-Stream-Verzweigungen wurden bewusst vermieden: upstream schaltet `STRATA_DF_BRANCH` bei Linux Open Modules ab (#905).
- Die Conv-Commits und die Indexer-Appends laufen weiterhin einmal pro Anfrage. Sie zu gruppieren ist der nächste Schritt.

## Lizenz

MIT. Die Patches verändern [Strata](https://github.com/Niko1221/Strata) (MIT, © 2026 Niko1221 und die
Strata-Mitwirkenden). Siehe [LICENSE](LICENSE).
