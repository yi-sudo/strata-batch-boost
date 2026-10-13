<h1 align="center">strata-batch-boost</h1>

[English](README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · **Português**

<p align="center"><b>Até +22% de velocidade total para requisições paralelas no <a href="https://github.com/Niko1221/Strata">Strata</a></b><br>
Mesmas saídas, byte a byte · velocidade de uma única requisição inalterada · duas variáveis de ambiente, desligadas por padrão</p>

O [Strata](https://github.com/Niko1221/Strata) roda o Qwen3.8-Flash-Next em um único PC e é muito rápido com **uma**
requisição por vez. Com várias requisições simultâneas (`--batch N`), cada requisição fica mais lenta do que deveria.
Este patch remove a maior parte desse overhead. MTP, a atenção QSA e o cache de experts na VRAM continuam exatamente
como estão.

| | Strata (upstream) | + este patch |
|---|---|---|
| 1 requisição | 179 tok/s | 179 tok/s |
| 4 requisições ao mesmo tempo (`--batch 4`), total | 268 tok/s | **312 tok/s** (+16%) |
| 8 requisições ao mesmo tempo (`--batch 8`), total | 308 tok/s | **375 tok/s** (+22%) |

Medido no Strata main `a0116b2` (2026-10-12) com uma RTX PRO 6000 Blackwell e Qwen3.8-Flash-Next IQ3_S. O Strata
0.1.43 mostra o mesmo quadro (veja [Resultados](#resultados)).

## Instalação

```bash
git clone https://github.com/yi-sudo/strata-batch-boost
./strata-batch-boost/apply.sh /path/to/Strata     # your Strata git clone
cd /path/to/Strata && ./setup.sh --build --no-start   # compile the engine from source
```

Depois ative-o no `strata-<model>.json` do seu modelo (o setup mantém este bloco quando reescreve o arquivo) e defina
quantas requisições rodam ao mesmo tempo:

```json
"parallel": 4,
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

Requer uma GPU NVIDIA e um CUDA toolkit para compilar (o patch altera kernels CUDA). O `./update.sh` do Strata faz um
`git pull`: rode `git checkout -- .` no Strata primeiro, atualize e depois rode `apply.sh` de novo.

## Por que as requisições paralelas eram lentas

A cada passo de decodificação, o Strata coloca o próximo token de cada requisição em andamento em uma única "janela de
verificação" (até 8 linhas):

| | tempo por janela | tokens por janela |
|---|---|---|
| 1 requisição + MTP | ~6.8 ms | ~1.56 (histórias), mais para código |
| `--batch` upstream, R linhas | ~4.8 ms + **~2.4 ms por linha** | R |

Dos 2.4 ms por linha, ler os experts leva só ~0.42 ms, o que já é o limite da largura de banda de memória. Quase todo o
resto são **pequenos lançamentos na GPU que rodam uma requisição após a outra**: a pontuação de blocos QSA, top-k e a
atenção, e a recorrência GDN, tanto na janela quanto no commit.

## O que o patch muda

| variável de ambiente | o que faz |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | Pontuação de blocos QSA, top-k e atenção de decodificação para **todas as linhas em um único lançamento cada**. Cada linha continua lendo as chaves do indexador e os pools K/V da sua própria requisição, que são passados por valor como parâmetros do kernel, de modo que o CUDA graph capturado para cada layout de linhas continua válido. Volta a lançamentos por linha quando uma requisição faz streaming do seu KV (`kv_mode 1`). |
| `STRATA_BATCH_GDN_GROUPS=1` | Recorrência GDN na janela, e o commit de 1 linha, para **todas as requisições em um único lançamento** (`grid.y` = requisição). |

A aritmética por linha é a mesma, então as saídas com temperatura 0 são idênticas byte a byte. Sem nenhuma das duas
variáveis definida, o motor se comporta exatamente como o upstream. Arquivos:
`src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`, seus headers e `src/core/verify.cpp` (+202 / −6
linhas).

## Configuração de teste

- NVIDIA RTX PRO 6000 Blackwell Workstation (96 GB), driver 610.57.04, CUDA 13.0, Linux
- Qwen3.8-Flash-Next IQ3_S, todos os 24,576 experts residentes na VRAM
- `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp --vision`, `--vram-reserve-mib 20480` salvo
  indicação em contrário
- Carga: [`bench_concurrency.py`](lab/bench_concurrency.py) envia N requisições em streaming ao mesmo tempo. Cada
  requisição recebe um prompt de história diferente, então nenhum prefixo é compartilhado. Cada uma gera 1024 tokens,
  com 2 execuções por ponto.

## Resultados

### Strata main `a0116b2` (a mais recente testada)

tok/s totais. As execuções com `--batch 8` usam uma reserva de 6144 MiB para que todos os experts ainda caibam na
VRAM. Dados: [latest-a0116b2](lab/results/latest-a0116b2).

| requisições | upstream `--batch 4` | com patch `--batch 4` | upstream `--batch 8` | com patch `--batch 8` |
|---|---|---|---|---|
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 ¹ | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | **311.7** | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | **374.8** |

¹ Repetida. A primeira execução deste ponto sofreu interferência (125.1, veja [NOTE](lab/results/latest-a0116b2/NOTE.md)).

Saídas com temperatura 0: `IDENTICAL` ao upstream tanto com `--batch 4` quanto com `--batch 8`.

### Strata 0.1.43 (`7695859`)

tok/s totais com `--batch 4` ([phase4](lab/results/phase4), [phase5](lab/results/phase5)):

| requisições | upstream | só atenção | atenção + GDN |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

tok/s totais com `--batch 8` ([phase6](lab/results/phase6)). Aqui a reserva é de 6144 MiB, então todos os experts
ainda cabem na VRAM:

| requisições | upstream | com patch |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

Tempo da janela de verificação em ms, a partir das próprias estatísticas `strata batch:` do motor:

| linhas | upstream | só atenção | atenção + GDN |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6 | 19.25 | | 15.92 |
| 8 | 24.55 | | 19.91 |

### Corretude

[`parity.py`](lab/parity.py) envia 4 prompts concorrentes com temperatura 0. Em seguida,
[`compare_parity.py`](lab/compare_parity.py) compara byte a byte as saídas com patch com as do upstream. Todas as
configurações testadas deram `IDENTICAL`.

### Linha de base do upstream: `--batch` e `--batch-mtp`

tok/s totais no upstream 0.1.43 com uma reserva de 20480 MiB ([production-p1-p4-p8](lab/results/production-p1-p4-p8/summary.txt)).
Com `--batch 8` só cabem na VRAM 17,097 dos 24,576 experts, por isso uma única requisição é mais lenta nesse caso.

| requisições | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### Memória: `--batch` reserva seus slots na inicialização

Dados: [context-memory](lab/results/context-memory/summary.txt).

A VRAM não cresce enquanto as requisições rodam. Ela ficou em 73,352 MiB antes, durante e depois de 4 requisições
concorrentes. Cada slot reserva espaço para o `--max-context` completo, então o tamanho do contexto determina quanto
custa o `--batch 4`:

| configuração | GPU inteira | 1 requisição | 4 requisições (512 tokens) |
|---|---|---|---|
| p1, 262K | 59.9 GB | 186 | 170 (em fila) |
| p4, 64K | 61.1 GB | 181 | 286 |
| p4, 128K | 65.7 GB | 181 | 291 |
| p4, 262K | ~74.9 GB | 184 | ~309 |

### Uma ideia que não compensou: mais rascunhos MTP por requisição

Dados: [phase3a](lab/results/phase3a), [phase3a2](lab/results/phase3a2), patch:
[`patches/experimental`](patches/experimental).

Este patch permite que cada requisição encadeie até D rascunhos MTP na janela. No modo batch só **50–57%** dos
rascunhos são aceitos, mas uma linha extra só compensa acima de ~67%. Com 4 requisições a velocidade total foi de 246
tok/s com D=1, 199 com D=3 e 193 com D=5, contra 268 do `--batch 4` sem MTP. O patch é mantido apenas como referência.

## Reproduzir

[`lab/`](lab) contém todos os scripts por trás destes números. O JSON bruto e o log do motor de cada execução estão em
[`lab/results/`](lab/results).

Defina `ROOT` (o clone do Strata com o patch), `STRATA_DATA` e `PY` em `lab/env.sh`, ou em um
`lab/env.local.sh` não rastreado pelo git. `lab-start.sh <parallel>` inicia um servidor de teste na porta 8095, e
`phaseN.sh` roda um experimento e para o servidor de novo.

| script | mede |
|---|---|
| `phase1.sh` | linha de base p1 vs p4 (a execução p1, 57 tok/s, é inválida: outra instância ainda estava na GPU) |
| `phase2.sh` | tempo da janela por etapa (`STRATA_DECODE_TIMING`, `STRATA_VERIFY_PROFILE`) |
| `phase3a.sh`, `phase3a2.sh` | o experimento de MTP com múltiplos rascunhos |
| `phase4.sh` | só atenção vs upstream, com paridade |
| `phase5.sh` | atenção + GDN, com paridade |
| `phase6.sh` | `--batch 8`, upstream vs com patch |
| `bench-dashboard.sh`, `context-memory.sh` | reinício por meio de um dashboard em várias configurações |

## Limitações

- Testado em uma GPU e um modelo: uma única RTX PRO 6000 com IQ3_S e todos os experts na VRAM. Não foi testado com
  várias GPUs, com experts via PCIe ou na CPU, nem em AMD.
- Ramificações multi-stream foram evitadas de propósito: o upstream desliga `STRATA_DF_BRANCH` nos módulos abertos do
  Linux (#905).
- Os commits de conv e os appends do indexador ainda rodam uma vez por requisição. Agrupá-los é o próximo passo.

## Licença

MIT. Os patches modificam o [Strata](https://github.com/Niko1221/Strata) (MIT, © 2026 Niko1221 e os colaboradores do
Strata). Veja [LICENSE](LICENSE).
