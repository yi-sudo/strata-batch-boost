<h1 align="center">strata-batch-boost</h1>

[English](README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Deutsch](README.de.md) · **Français** · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Jusqu'à +22 % de vitesse totale pour les requêtes parallèles dans <a href="https://github.com/Niko1221/Strata">Strata</a></b><br>
Sorties identiques octet par octet · vitesse d'une requête seule inchangée · deux variables d'environnement, désactivées par défaut</p>

[Strata](https://github.com/Niko1221/Strata) exécute Qwen3.8-Flash-Next sur un seul PC et se montre très rapide pour
**une** requête à la fois. Avec plusieurs requêtes simultanées (`--batch N`), chaque requête ralentit plus qu'elle ne le
devrait. Ce patch supprime l'essentiel de ce surcoût. MTP, l'attention QSA et le cache d'experts en VRAM restent
exactement tels quels.

| | Strata (upstream) | + ce patch |
|---|---|---|
| 1 requête | 179 tok/s | 179 tok/s |
| 4 requêtes simultanées (`--batch 4`), total | 268 tok/s | **312 tok/s** (+16 %) |
| 8 requêtes simultanées (`--batch 8`), total | 308 tok/s | **375 tok/s** (+22 %) |

Mesuré sur Strata main `a0116b2` (2026-10-12) avec une RTX PRO 6000 Blackwell et Qwen3.8-Flash-Next IQ3_S. Strata
0.1.43 donne la même tendance (voir [Résultats](#résultats)).

## Installation

```bash
git clone https://github.com/yi-sudo/strata-batch-boost
./strata-batch-boost/apply.sh /path/to/Strata     # your Strata git clone
cd /path/to/Strata && ./setup.sh --build --no-start   # compile the engine from source
```

Activez-le ensuite dans le fichier `strata-<model>.json` de votre modèle (le setup conserve ce bloc lorsqu'il réécrit le
fichier) et indiquez combien de requêtes s'exécutent en même temps :

```json
"parallel": 4,
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

La compilation nécessite un GPU NVIDIA et un CUDA toolkit (le patch modifie des kernels CUDA). Le `./update.sh` de
Strata fait un `git pull` : lancez d'abord `git checkout -- .` dans Strata, mettez à jour, puis relancez `apply.sh`.

## Pourquoi les requêtes parallèles étaient lentes

À chaque étape de décodage, Strata place le prochain token de chaque requête en cours dans une même « fenêtre de
vérification » (jusqu'à 8 lignes) :

| | temps par fenêtre | tokens par fenêtre |
|---|---|---|
| 1 requête + MTP | ~6,8 ms | ~1,56 (histoires), davantage pour du code |
| upstream `--batch`, R lignes | ~4,8 ms + **~2,4 ms par ligne** | R |

Sur les 2,4 ms par ligne, la lecture des experts ne prend que ~0,42 ms, ce qui atteint déjà la limite de bande passante
mémoire. L'essentiel du reste correspond à **de petits lancements GPU exécutés une requête après l'autre** : le scoring
des blocs QSA, le top-k et l'attention, ainsi que la récurrence GDN, à la fois dans la fenêtre et lors du commit.

## Ce que change le patch

| variable d'env. | effet |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | Scores des blocs QSA, top-k et attention de décodage pour **toutes les lignes en un seul lancement chacun**. Chaque ligne lit toujours les clés d'indexeur et les pools K/V de sa propre requête, passés par valeur comme paramètres du kernel, de sorte que le CUDA graph capturé pour chaque disposition de lignes reste valide. Revient à des lancements par ligne quand une requête streame son KV (`kv_mode 1`). |
| `STRATA_BATCH_GDN_GROUPS=1` | Récurrence GDN dans la fenêtre, et le commit sur 1 ligne, pour **toutes les requêtes en un seul lancement** (`grid.y` = requête). |

L'arithmétique par ligne est la même, donc les sorties à température 0 sont identiques octet par octet. Sans aucune des
deux variables, le moteur se comporte exactement comme upstream. Fichiers :
`src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`, leurs en-têtes et `src/core/verify.cpp`
(+202 / −6 lignes).

## Configuration de test

- NVIDIA RTX PRO 6000 Blackwell Workstation (96 Go), pilote 610.57.04, CUDA 13.0, Linux
- Qwen3.8-Flash-Next IQ3_S, les 24 576 experts résidents en VRAM
- `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp --vision`, `--vram-reserve-mib 20480` sauf indication contraire
- Charge : [`bench_concurrency.py`](lab/bench_concurrency.py) envoie N requêtes en streaming au même moment. Chaque
  requête reçoit un prompt d'histoire différent, donc aucun préfixe n'est partagé. Chacune produit 1024 tokens, avec 2
  exécutions par point.

## Résultats

### Strata main `a0116b2` (version testée la plus récente)

Total en tok/s. Les exécutions `--batch 8` utilisent une réserve de 6144 MiB pour que tous les experts tiennent encore en
VRAM. Données : [latest-a0116b2](lab/results/latest-a0116b2).

| requêtes | upstream `--batch 4` | patché `--batch 4` | upstream `--batch 8` | patché `--batch 8` |
|---|---|---|---|---|
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 ¹ | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | **311.7** | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | **374.8** |

¹ Relancé. La première exécution de ce point avait été perturbée (125.1, voir [NOTE](lab/results/latest-a0116b2/NOTE.md)).

Sorties à température 0 : `IDENTICAL` à upstream pour `--batch 4` comme pour `--batch 8`.

### Strata 0.1.43 (`7695859`)

Total en tok/s avec `--batch 4` ([phase4](lab/results/phase4), [phase5](lab/results/phase5)) :

| requêtes | upstream | attention seule | attention + GDN |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

Total en tok/s avec `--batch 8` ([phase6](lab/results/phase6)). La réserve est ici de 6144 MiB, pour que tous les
experts tiennent encore en VRAM :

| requêtes | upstream | patché |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

Temps de la fenêtre de vérification en ms, d'après les statistiques `strata batch:` du moteur lui-même :

| lignes | upstream | attention seule | attention + GDN |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6 | 19.25 | | 15.92 |
| 8 | 24.55 | | 19.91 |

### Exactitude

[`parity.py`](lab/parity.py) envoie 4 prompts simultanés à température 0. [`compare_parity.py`](lab/compare_parity.py)
compare ensuite les sorties patchées à celles d'upstream, octet par octet. Toutes les configurations testées ont donné
`IDENTICAL`.

### Référence upstream : `--batch` et `--batch-mtp`

Total en tok/s sur upstream 0.1.43 avec une réserve de 20480 MiB ([production-p1-p4-p8](lab/results/production-p1-p4-p8/summary.txt)).
Avec `--batch 8`, seuls 17 097 des 24 576 experts tiennent en VRAM, ce qui explique pourquoi une requête seule y est plus lente.

| requêtes | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### Mémoire : `--batch` réserve ses slots au démarrage

Données : [context-memory](lab/results/context-memory/summary.txt).

La VRAM n'augmente pas pendant l'exécution des requêtes. Elle est restée à 73 352 MiB avant, pendant et après 4 requêtes
simultanées. Chaque slot réserve de la place pour le `--max-context` complet, donc c'est la taille de contexte qui
détermine ce que coûte `--batch 4` :

| réglage | GPU entier | 1 requête | 4 requêtes (512 tokens) |
|---|---|---|---|
| p1, 262K | 59.9 GB | 186 | 170 (en file d'attente) |
| p4, 64K | 61.1 GB | 181 | 286 |
| p4, 128K | 65.7 GB | 181 | 291 |
| p4, 262K | ~74.9 GB | 184 | ~309 |

### Une idée qui n'a pas payé : plus de drafts MTP par requête

Données : [phase3a](lab/results/phase3a), [phase3a2](lab/results/phase3a2), patch :
[`patches/experimental`](patches/experimental).

Ce patch permet à chaque requête d'enchaîner jusqu'à D drafts MTP dans la fenêtre. En mode batch, seuls **50–57 %** des
drafts sont acceptés, alors qu'une ligne supplémentaire n'est rentable qu'au-delà de ~67 %. Avec 4 requêtes, la vitesse
totale était de 246 tok/s à D=1, 199 à D=3 et 193 à D=5, contre 268 pour `--batch 4` sans MTP. Le patch n'est conservé
qu'à titre de référence.

## Reproduire

[`lab/`](lab) contient tous les scripts derrière ces chiffres. Le JSON brut et le log du moteur de chaque exécution se
trouvent dans [`lab/results/`](lab/results).

Définissez `ROOT` (le clone Strata patché), `STRATA_DATA` et `PY` dans `lab/env.sh`, ou dans un fichier non suivi
`lab/env.local.sh`. `lab-start.sh <parallel>` démarre un serveur de test sur le port 8095, et `phaseN.sh` exécute une
expérience puis arrête de nouveau le serveur.

| script | mesure |
|---|---|
| `phase1.sh` | référence p1 vs p4 (son exécution p1, à 57 tok/s, n'est pas valide : une autre instance tournait encore sur le GPU) |
| `phase2.sh` | temps de fenêtre par étape (`STRATA_DECODE_TIMING`, `STRATA_VERIFY_PROFILE`) |
| `phase3a.sh`, `phase3a2.sh` | l'expérience MTP multi-drafts |
| `phase4.sh` | attention seule vs upstream, avec parité |
| `phase5.sh` | attention + GDN, avec parité |
| `phase6.sh` | `--batch 8`, upstream vs patché |
| `bench-dashboard.sh`, `context-memory.sh` | redémarrage via un dashboard avec plusieurs réglages |

## Limites

- Testé sur un seul GPU et un seul modèle : une RTX PRO 6000 avec IQ3_S et tous les experts en VRAM. Non testé avec
  plusieurs GPU, avec des experts via PCIe ou sur le CPU, ni sur AMD.
- Les branches multi-stream ont été évitées volontairement : upstream désactive `STRATA_DF_BRANCH` sur les modules open
  Linux (#905).
- Les commits conv et les ajouts de l'indexeur s'exécutent encore une fois par requête. Les regrouper est la prochaine étape.

## Licence

MIT. Les patches modifient [Strata](https://github.com/Niko1221/Strata) (MIT, © 2026 Niko1221 et les contributeurs de
Strata). Voir [LICENSE](LICENSE).
