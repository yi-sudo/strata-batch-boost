<h1 align="center">strata-batch-boost</h1>

[English](README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · **Español** · [Português](README.pt-BR.md)

<p align="center"><b>Hasta +22% de velocidad total para peticiones paralelas en <a href="https://github.com/Niko1221/Strata">Strata</a></b><br>
Mismas salidas, byte a byte · velocidad de una sola petición sin cambios · dos variables de entorno, desactivadas por defecto</p>

[Strata](https://github.com/Niko1221/Strata) ejecuta Qwen3.8-Flash-Next en un solo PC y es muy rápido con **una**
petición a la vez. Con varias peticiones simultáneas (`--batch N`), cada petición se ralentiza más de lo debido. Este
parche elimina la mayor parte de esa sobrecarga. MTP, la atención QSA y la caché de expertos en VRAM quedan exactamente
igual.

| | Strata (upstream) | + este parche |
|---|---|---|
| 1 petición | 179 tok/s | 179 tok/s |
| 4 peticiones a la vez (`--batch 4`), total | 268 tok/s | **312 tok/s** (+16%) |
| 8 peticiones a la vez (`--batch 8`), total | 308 tok/s | **375 tok/s** (+22%) |

Medido en Strata main `a0116b2` (2026-10-12) con una RTX PRO 6000 Blackwell y Qwen3.8-Flash-Next IQ3_S. Strata
0.1.43 da el mismo panorama (ver [Resultados](#resultados)).

## Instalación

```bash
git clone https://github.com/yi-sudo/strata-batch-boost
./strata-batch-boost/apply.sh /path/to/Strata     # your Strata git clone
cd /path/to/Strata && ./setup.sh --build --no-start   # compile the engine from source
```

Después actívalo en el `strata-<model>.json` de tu modelo (setup conserva este bloque cuando reescribe el archivo) y
define cuántas peticiones se ejecutan a la vez:

```json
"parallel": 4,
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

Requiere una GPU NVIDIA y un CUDA toolkit para compilar (el parche modifica kernels CUDA). El `./update.sh` de Strata
hace un `git pull`: ejecuta primero `git checkout -- .` en Strata, actualiza y luego vuelve a ejecutar `apply.sh`.

## Por qué las peticiones paralelas eran lentas

En cada paso de decodificación, Strata coloca el siguiente token de cada petición en curso en una única "ventana de
verificación" (hasta 8 filas):

| | tiempo por ventana | tokens por ventana |
|---|---|---|
| 1 petición + MTP | ~6.8 ms | ~1.56 (historias), más para código |
| `--batch` upstream, R filas | ~4.8 ms + **~2.4 ms por fila** | R |

De los 2.4 ms por fila, leer los expertos lleva solo ~0.42 ms, que ya es el límite del ancho de banda de memoria. Casi
todo el resto son **pequeños lanzamientos en la GPU que se ejecutan una petición tras otra**: la puntuación de bloques
QSA, top-k y la atención, y la recurrencia GDN, tanto en la ventana como al hacer commit.

## Qué cambia el parche

| variable de entorno | qué hace |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | Puntuación de bloques QSA, top-k y atención de decodificación para **todas las filas en un solo lanzamiento cada una**. Cada fila sigue leyendo las claves del indexador y los pools K/V de su propia petición, que se pasan por valor como parámetros del kernel, de modo que el CUDA graph capturado por cada disposición de filas sigue siendo válido. Vuelve a lanzamientos por fila cuando una petición hace streaming de su KV (`kv_mode 1`). |
| `STRATA_BATCH_GDN_GROUPS=1` | Recurrencia GDN en la ventana, y el commit de 1 fila, para **todas las peticiones en un solo lanzamiento** (`grid.y` = petición). |

La aritmética por fila es la misma, así que las salidas a temperatura 0 son idénticas byte a byte. Sin ninguna de las
dos variables, el motor se comporta exactamente igual que upstream. Archivos:
`src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`, sus cabeceras y `src/core/verify.cpp` (+202 / −6
líneas).

## Configuración de prueba

- NVIDIA RTX PRO 6000 Blackwell Workstation (96 GB), driver 610.57.04, CUDA 13.0, Linux
- Qwen3.8-Flash-Next IQ3_S, los 24,576 expertos residentes en VRAM
- `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp --vision`, `--vram-reserve-mib 20480` salvo que se
  indique otra cosa
- Carga: [`bench_concurrency.py`](lab/bench_concurrency.py) envía N peticiones en streaming al mismo tiempo. Cada
  petición recibe un prompt de historia distinto, así que no se comparte ningún prefijo. Cada una genera 1024 tokens,
  con 2 ejecuciones por punto.

## Resultados

### Strata main `a0116b2` (la más reciente probada)

tok/s totales. Las ejecuciones con `--batch 8` usan una reserva de 6144 MiB para que todos los expertos sigan cabiendo
en VRAM. Datos: [latest-a0116b2](lab/results/latest-a0116b2).

| peticiones | upstream `--batch 4` | con parche `--batch 4` | upstream `--batch 8` | con parche `--batch 8` |
|---|---|---|---|---|
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 ¹ | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | **311.7** | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | **374.8** |

¹ Repetida. La primera ejecución de este punto se vio perturbada (125.1, ver [NOTE](lab/results/latest-a0116b2/NOTE.md)).

Salidas a temperatura 0: `IDENTICAL` a upstream tanto con `--batch 4` como con `--batch 8`.

### Strata 0.1.43 (`7695859`)

tok/s totales con `--batch 4` ([phase4](lab/results/phase4), [phase5](lab/results/phase5)):

| peticiones | upstream | solo atención | atención + GDN |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

tok/s totales con `--batch 8` ([phase6](lab/results/phase6)). Aquí la reserva es de 6144 MiB, así que todos los
expertos siguen cabiendo en VRAM:

| peticiones | upstream | con parche |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

Tiempo de la ventana de verificación en ms, según las propias estadísticas `strata batch:` del motor:

| filas | upstream | solo atención | atención + GDN |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6 | 19.25 | | 15.92 |
| 8 | 24.55 | | 19.91 |

### Corrección

[`parity.py`](lab/parity.py) envía 4 prompts concurrentes a temperatura 0. Después,
[`compare_parity.py`](lab/compare_parity.py) compara byte a byte las salidas con parche con las de upstream. Todas las
configuraciones probadas dieron `IDENTICAL`.

### Línea base de upstream: `--batch` y `--batch-mtp`

tok/s totales en upstream 0.1.43 con una reserva de 20480 MiB ([production-p1-p4-p8](lab/results/production-p1-p4-p8/summary.txt)).
Con `--batch 8` solo caben en VRAM 17,097 de los 24,576 expertos, por eso ahí una sola petición es más lenta.

| peticiones | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### Memoria: `--batch` reserva sus slots al arrancar

Datos: [context-memory](lab/results/context-memory/summary.txt).

La VRAM no crece mientras se ejecutan las peticiones. Se mantuvo en 73,352 MiB antes, durante y después de 4
peticiones concurrentes. Cada slot reserva espacio para el `--max-context` completo, así que el tamaño de contexto
determina lo que cuesta `--batch 4`:

| ajuste | GPU completa | 1 petición | 4 peticiones (512 tokens) |
|---|---|---|---|
| p1, 262K | 59.9 GB | 186 | 170 (en cola) |
| p4, 64K | 61.1 GB | 181 | 286 |
| p4, 128K | 65.7 GB | 181 | 291 |
| p4, 262K | ~74.9 GB | 184 | ~309 |

### Una idea que no compensó: más borradores MTP por petición

Datos: [phase3a](lab/results/phase3a), [phase3a2](lab/results/phase3a2), parche:
[`patches/experimental`](patches/experimental).

Este parche permite que cada petición encadene hasta D borradores MTP en la ventana. En modo batch solo se acepta el
**50–57%** de los borradores, pero una fila extra solo compensa por encima de ~67%. Con 4 peticiones la velocidad
total fue de 246 tok/s con D=1, 199 con D=3 y 193 con D=5, frente a 268 de `--batch 4` sin MTP. El parche se conserva
solo como referencia.

## Reproducir

[`lab/`](lab) contiene todos los scripts detrás de estas cifras. El JSON en bruto y el log del motor de cada ejecución
están en [`lab/results/`](lab/results).

Define `ROOT` (el clon de Strata con el parche), `STRATA_DATA` y `PY` en `lab/env.sh`, o en un
`lab/env.local.sh` sin seguimiento en git. `lab-start.sh <parallel>` inicia un servidor de pruebas en el puerto 8095, y
`phaseN.sh` ejecuta un experimento y vuelve a detener el servidor.

| script | mide |
|---|---|
| `phase1.sh` | línea base p1 vs p4 (su ejecución p1, 57 tok/s, no es válida: otra instancia seguía en la GPU) |
| `phase2.sh` | tiempo de ventana por etapa (`STRATA_DECODE_TIMING`, `STRATA_VERIFY_PROFILE`) |
| `phase3a.sh`, `phase3a2.sh` | el experimento de MTP con varios borradores |
| `phase4.sh` | solo atención vs upstream, con paridad |
| `phase5.sh` | atención + GDN, con paridad |
| `phase6.sh` | `--batch 8`, upstream vs con parche |
| `bench-dashboard.sh`, `context-memory.sh` | reinicio a través de un dashboard con varios ajustes |

## Limitaciones

- Probado en una GPU y un modelo: una sola RTX PRO 6000 con IQ3_S y todos los expertos en VRAM. No se ha probado con
  varias GPU, con expertos en PCIe o en la CPU, ni en AMD.
- Se evitaron a propósito las ramas multi-stream: upstream desactiva `STRATA_DF_BRANCH` en los módulos abiertos de
  Linux (#905).
- Los commits de conv y los appends del indexador siguen ejecutándose una vez por petición. Agruparlos es el siguiente
  paso.

## Licencia

MIT. Los parches modifican [Strata](https://github.com/Niko1221/Strata) (MIT, © 2026 Niko1221 y los colaboradores de
Strata). Ver [LICENSE](LICENSE).
