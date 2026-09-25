# vidinfer

🇬🇧 [English version](README.md)

Décoder une vidéo, lancer une inférence à **10 inférences par seconde de vidéo** sur des images **RGB 720×1280**, et
loguer **la performance toutes les 10 inférences**, avec des résultats dont la reproductibilité est vérifiée, pas
supposée.

Modèle : détection des personnes et du ballon (poids officiels Ultralytics YOLO26s, classes COCO `person` et
`sports ball`).

## Lecture de l'énoncé

- « Chacune des images » et « 10 inférences par seconde de vidéo » ne tiennent ensemble que pour les images
  **échantillonnées** : 300 s de vidéo → **3 000 inférences**, une ligne de sortie chacune.
- « Toutes les 10 images » → une ligne par bloc de **10 inférences = 1 s de vidéo**, avec le temps passé dans chaque étape.
- « Vérifier / garantir le RGB » → RGB **par construction** (un seul appel de conversion) **et vérifié** (contrat de
  forme et de type sur chaque image).
- « 720 (h) × 1280 (l) » → même ratio 16:9 que l'entrée 1920×1080 : redimensionnement sans déformation.

## Résultats sur `cut.mp4` (run de référence, `results/reference/`, commit `3600844`)

| | |
|---|---|
| Entrée | H.264 1920×1080, 25 fps constants, 7 500 images, 300 s, espace couleur non déclaré |
| Statut | `complete` : **3 000 / 3 000** inférences, 7 500 / 7 500 images décodées, 0 erreur de décodage, écart d'échantillonnage max 20 ms |
| Débit | **64,6 inférences/s = ×6,5 le temps réel** pour la boucle de traitement (46,5 s pour 300 s) ; ×5,9 pour le programme entier, chargement et préchauffage du modèle compris |
| Temps de service par image | p50 15,0 ms · p95 17,1 ms · p99 18,5 ms (conversion + inférence + écriture ; le décodage tourne dans les threads FFmpeg) |
| Répartition du temps | inférence 69 % · prétraitement 25 % · empreinte 4 % · attente décodage 2 % · écriture 0,6 % |
| GPU | RTX 4050 Laptop ; le réseau tourne 52 % du temps réel ; pic de 144 Mio alloués par torch |
| Reproductibilité | référence, second run et run Docker au même commit : `results_sha256` et `frames_crc32` identiques |

`results/reference/viz/` : une frise reconstruite depuis `run.log`, et 4 images annotées : deux réussites (t=20 s,
t=240 s) et deux échecs (t=8,3 s : des panneaux publicitaires détectés comme ballons ; t=36 s : gros plan flou, rien
de détecté).

![frise](results/reference/viz/timeline.png)

## Prérequis

- **Linux x86_64 + GPU NVIDIA** avec un driver compatible CUDA 12.6 (testé avec 580). Le fichier de verrouillage ne
  cible que cette plateforme : sous macOS ou Windows, lire `results/reference/`, ou utiliser Docker sur un hôte Linux
  avec GPU.
- Sans GPU : `--device cpu` tourne à ≈ 2 inférences/s (≈ 25 min pour la vidéo complète ; utiliser `--max-frames 100`).
- La vidéo n'est pas dans le dépôt : la placer en `data/cut.mp4` (sha256 `4072bd68753d20120615054f9fabfd0d09e67e765e93b9fcd717ab5748405eb7`).
- Disque : ≈ 7 Go pour l'environnement (wheels CUDA) ou pour l'image. Le premier run local télécharge les poids figés
  (20 Mo) ; l'image Docker les contient déjà.

## Démarrage rapide

```bash
uv sync --locked                                              # environnement exact depuis uv.lock
uv run python -m vidinfer data/cut.mp4 --output-dir outputs   # -> outputs/output.jsonl, outputs/run.log
uv run python -m vidinfer.viz outputs                         # -> outputs/viz/*.png|jpg
uv run pytest                                                 # 29 tests, CPU seul, ~8 s, vidéos synthétiques
```

Docker (GPU via le NVIDIA Container Toolkit ; poids YOLO26s intégrés à l'image et vérifiés au build) :

```bash
docker build --build-arg GIT_COMMIT=$(git describe --always --dirty) -t vidinfer:0.2.0 .
mkdir -p outputs
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/data:ro" -v "$PWD/outputs:/work/outputs" vidinfer:0.2.0 /data/cut.mp4
```

Options : `--fps 10`, `--model yolo26s.pt|yolo26m.pt|chemin.pt` (seul `s` est intégré à l'image ; pour `m`, monter
un dossier de poids sur `VIDINFER_WEIGHTS_DIR`), `--device auto|cuda|cpu`, `--max-frames N` (test rapide).

| Code de sortie | Signification | Action de l'orchestrateur |
|---|---|---|
| 0 | `complete`, ou `stopped_early` avec `--max-frames` | terminé |
| 1 | erreur inattendue (ex. panne CUDA) | relancer |
| 2 | entrée ou configuration invalide (vidéo absente ou illisible, argument invalide, modèle inconnu, sha256 des poids non conforme) | ne pas relancer, corriger l'entrée |
| 3 | `partial` : erreurs de décodage ou timestamps cassés, la vidéo n'a pas été traitée entièrement | ne pas marquer comme traitée, revoir |

## Exigence → implémentation → preuve

| Exigence | Implémentation | Preuve |
|---|---|---|
| Programme Python prenant `cut.mp4` | `python -m vidinfer VIDEO` | `tests/test_cli.py`, de bout en bout sur des vidéos synthétiques |
| Redimensionner en 720×1280 | un seul appel swscale : YUV→RGB + redimensionnement `AREA` ; WARNING si l'entrée n'est pas en 16:9 | contrat vérifié sur chaque image |
| Garantir le RGB | `rgb24` par construction ; HD non déclarée → BT.709 (les bibliothèques appliquent BT.601 par défaut), signalé en WARNING ; BGR uniquement à l'appel Ultralytics | tests : vidéo rouge pur, règle de matrice, BT.709 appliqué sur une vidéo 1280×720 non déclarée, BGR à la frontière du modèle |
| 10 inférences par seconde de vidéo | image la plus proche de `t0 + k/10`, égalité → image antérieure, fractions exactes | `3000/3000` et écart max de 20 ms dans le résumé ; `test_sampler.py` ; une vidéo dont chaque image porte son numéro dans ses pixels |
| Loguer le temps toutes les 10 images | une ligne `event=block` toutes les 10 inférences, avec le détail par étape | 300 lignes de bloc dans `run.log`, relues par les tests et par `viz` |
| Temps total + métadonnées | `event=timing` (programme, préparation, chargement du modèle, préchauffage, traitement, CPU) et `event=video` (codec, taille, fps, durée, images, tags couleur, ratio d'affichage, sha256) | `run.log`, en-tête de `output.jsonl` |
| `output.<format>` pour chaque image | `output.jsonl` : en-tête / une ligne par image inférée (même sans détection) / résumé avec `status` | `results/reference/output.jsonl` |
| Images ou visualisation | `python -m vidinfer.viz` : frise depuis `run.log`, images annotées re-décodées et contrôlées contre `output.jsonl` | `results/reference/viz/` |

## Méthode et décisions

Chaque décision a suivi la même boucle : **lire l'énoncé comme une spécification → observer la donnée → lister ce qui
peut être faux sans bruit → mesurer les options sur la vraie vidéo → décider par une règle explicite → couvrir par un
test → s'arrêter** (ce qui n'est ni demandé, ni testé, ni bon marché part en roadmap, avec la condition qui le ferait
revenir).

| Décision | Options étudiées | Choix | Pourquoi |
|---|---|---|---|
| 25 → 10 fps | 1 image sur N, timestamps flottants, filtre `fps` de ffmpeg, *floor*, plus proche | image la plus proche, égalité → antérieure, `Fraction`, grille calée sur le premier PTS | 25/10 = 2,5 : une cible sur deux est une égalité exacte. Les flottants tranchent par bruit d'arrondi ; le `fps=10` par défaut de ffmpeg prend les images 1, 3, 6, 8… (20 à 40 ms de retard). « La plus proche » borne l'erreur à une demi-image. |
| Décodeur | OpenCV, pipe ffmpeg, PyAV, torchcodec | PyAV, `thread_type="AUTO"` | PTS exacts et matrice couleur maîtrisée ; décoder toutes les images H.264 est de toute façon obligatoire (P-frames) |
| Matrice couleur | défaut des bibliothèques / explicite | BT.709 pour la HD non déclarée, logué | une hypothèse (la convention HD usuelle), rendue visible : une mauvaise matrice fait passer un rouge pur de G=0 à G=23 (suite de tests) |
| Frontière du modèle | RGB tel quel / conversion | `cv2.cvtColor` vers BGR uniquement à l'appel Ultralytics | Ultralytics traite un tableau numpy comme du BGR (`engine/predictor.py`) ; passer du RGB est une erreur silencieuse |
| Modèle | YOLO26 n/s/m, imgsz 640/1280, FP16/32 | YOLO26s, imgsz 1280, FP16, conf 0,25 | le ballon fait 8 à 16 px en 720p, d'où imgsz 1280. s contre m est un arbitrage de **coût** : sur un échantillon étiqueté à la main, m retrouve tous les ballons de s plus 3 sur 27, pour 2,4× plus de temps GPU ; l'échantillon est trop petit pour conclure (mesures de l'étude, scripts hors de ce dépôt). m est à un paramètre près. |
| Boucle | threads producteur/consommateur, batching | boucle synchrone | déjà ×6,5 le temps réel ; le décodage tourne dans les threads FFmpeg (0,26 ms d'attente par image) |
| Sortie | JSON, CSV, Parquet, COCO, MOT | JSON Lines | streamable, lisible après un crash, garde les images sans détection ; le `status` du résumé marque la complétude |

## Reproductibilité

Chaque run se termine par deux empreintes :

- `results_sha256` : empreinte de `(index d'image source, détections)` pour chaque échantillon, temps exclus.
- `frames_crc32` : CRC de tous les pixels envoyés au modèle (0,55 ms/image, chronométré comme une étape à part). Si
  les résultats divergent, elle dit si l'écart vient du décodage/prétraitement ou du modèle.

`results/reproducibility.log` contient les lignes start, environnement, résumé, temps et empreintes de trois runs au
commit `3600844` : référence et second run dans le venv de l'hôte (Python 3.12.13, glibc 2.43), et l'image Docker
(Python 3.12.14, glibc 2.36). Les trois donnent `results_sha256 = 824865c8…a40a9c` et `frames_crc32 = 0fb35010`,
les mêmes valeurs qu'en version 0.1.0 : les corrections de la 0.2.0 n'ont changé aucune détection.

**Périmètre** : identique bit à bit avec le même modèle de GPU, le même driver et les mêmes wheels. Sur CPU,
`results_sha256` diffère (FP32 au lieu de FP16, autres kernels : boîtes à 0,7 px près) alors que `frames_crc32` reste
identique ; le même comportement est attendu sur un autre modèle de GPU. Les pixels sont reproductibles partout, les
détections au sein d'une même classe de matériel.

| Couche | Figée par | Enregistrée dans |
|---|---|---|
| Code | commit git (`-dirty` si `src/` ou les dépendances diffèrent) ; label d'image `org.opencontainers.image.revision` | `event=start`, en-tête de sortie |
| Dépendances Python | `uv.lock` (versions + hashes ; torch depuis l'index cu126) | `event=environment` (versions, CUDA, cuDNN, GPU, driver) |
| Image système | image de base par digest, uv par version (les paquets apt ne sont pas figés : construire une fois, déployer par digest) | Dockerfile |
| Poids du modèle | URL de release exacte + sha256, vérifié **avant** chargement (un `.pt` est un pickle) ; Docker `ADD --checksum` | `event=model`, en-tête de sortie |
| Entrée | sha256 de la vidéo | `event=video`, en-tête de sortie |
| Effets de bord à l'exécution | `YOLO_OFFLINE=1` (pas de télémétrie), `YOLO_AUTOINSTALL=0` (aucun pip install en cours de run) | — |
| Reconstruction sur machine vierge | CI : `uv sync --locked` + lint + tests à chaque push | `.github/workflows/ci.yml` |

Par défaut, Ultralytics télécharge les poids depuis la *dernière* release GitHub : d'où l'URL figée.

## Observabilité de la performance

Chaque ligne de log est au format **logfmt** avec un horodatage UTC, lisible par un humain et par n'importe quel
parseur logfmt (Loki, Vector…) :

```text
ts=2026-09-25T07:14:20.940Z level=INFO event=block block=150 samples=1490-1499 video_t_s=149.90 wall_ms=154.67 ms_per_frame=15.47 inf_per_s=64.65 speed=6.47 decode_ms=2.52 preprocess_ms=38.45 fingerprint_ms=5.29 inference_ms=107.17 write_ms=0.97 other_ms=0.26 yolo_preprocess_ms=10.38 yolo_inference_ms=80.82 yolo_postprocess_ms=8.24 cpu_cores=1.81 gpu_mem_mb=50.25 elapsed_s=23.00
```

- `speed` = secondes de vidéo traitées par seconde réelle (le SLO temps réel) ; `speed < 1` émet un WARNING.
- Les étapes somment à `wall_ms` ; `other_ms` est le reste non expliqué : rien n'est caché.
- Les étapes GPU sont chronométrées après `torch.cuda.synchronize()` ; `yolo_*_ms` sont les temps internes
  d'Ultralytics, gardés à côté des nôtres.
- `cpu_cores` = secondes CPU par seconde réelle (thread principal + threads de décodage FFmpeg) ; `gpu_mem_mb` =
  mémoire allouée par torch (`nvidia-smi` indique ≈ 320 Mio pour le processus, contexte CUDA compris).

Ce que les logs ont montré, et ce qui a changé grâce à eux :

1. **Mesurer à ma frontière a révélé mon propre coût.** L'appel au modèle mesurait 12,4 ms par image contre 9,1 ms
   annoncés par Ultralytics. L'écart de 3,3 ms venait de ma conversion RGB→BGR (`np.ascontiguousarray(rgb[..., ::-1])`,
   ≈ 5 ms isolée) : `cv2.cvtColor` produit les mêmes pixels en 0,07 ms. Écart ramené à 0,8 ms, débit ×5,8 → ×6,5,
   empreintes identiques.
2. **Le décodage ne coûte presque rien au thread principal** (0,26 ms d'attente par image) : les threads FFmpeg
   décodent en parallèle.
3. **Le GPU fait tourner le réseau environ la moitié du temps.** Prochaine cible si besoin : le prétraitement
   (3,9 ms/image, swscale mono-thread), ou alimenter le GPU depuis un thread de décodage. Inutile à ×6,5.

```python
import shlex
blocks = [dict(t.split("=", 1) for t in shlex.split(line.split(" event=block ", 1)[1]))
          for line in open("outputs/run.log") if " event=block " in line]
```

## Format de sortie (`output.jsonl`, schéma 1.1.0)

```json
{"type":"header","schema_version":"1.1.0","program":{"git_commit":"3600844…"},"environment":{…},"video":{"sha256":"4072bd68…",…},"processing":{"target_fps":10,"yuv_matrix":"ITU709",…},"model":{"weights":"yolo26s.pt","sha256":"646f8bc3…",…},"coordinates":"bbox_xyxy in pixels of the 1280x720 frame"}
{"type":"frame","sample_index":2400,"source_frame_index":6000,"pts":3072000,"timestamp_s":240.0,"inference_ms":…,"detections":[{"bbox_xyxy":[628.0,537.0,641.0,550.0],"score":0.8081,"class_id":32,"class_name":"sports ball"},…]}
{"type":"summary","status":"complete","n_inferences":3000,"expected":3000,"decode_errors":0,"max_sampling_error_ms":20.0,"duplicate_samples":0,"speed_x":6.46,…,"results_sha256":"824865c8…","frames_crc32":"0fb35010"}
```

La complétude, c'est `summary.status == "complete"`, pas la présence de la ligne : sur un fichier corrompu,
l'échantillonneur comble les trous avec l'image décodée la plus proche, donc le nombre d'inférences peut tomber juste
alors que des images ont été perdues (voir `test_corrupt_video_is_partial_not_success`). Les lecteurs ignorent les
clés inconnues.

## Limites

- **Licence** : Ultralytics est sous AGPL-3.0. Un usage en production demande une licence Enterprise ou un détecteur
  Apache-2.0 (RF-DETR a été envisagé pendant l'étude). Seul `model.py` changerait, plus les identifiants de classes
  utilisés par `viz`.
- **Aucune mesure de qualité** : l'énoncé n'en demande pas, et il n'y a pas de données annotées. Ce que montre la
  sortie : 13,9 personnes par image en moyenne ; 77 % des images ont au moins une détection `sports ball`, mais 29 %
  en ont deux ou plus, c'est-à-dire des faux ballons sur les panneaux et des ballons de réserve (`frame_t008.3s.jpg`).
  13 images n'ont aucune détection : 9 dans le générique d'ouverture, 4 dans des gros plans flous (`frame_t036.0s.jpg`).
- **BT.709 est une hypothèse** pour ce flux HD non déclaré ; la matrice est loguée et un flux déclaré est respecté.
- **Géométrie de l'entrée** : une entrée qui n'est pas en 16:9 est étirée (WARNING) ; l'entrelacement et les
  métadonnées de rotation ne sont pas gérés.
- Les temps viennent d'un seul GPU de portable et varient de quelques pour cent d'un run à l'autre (×6,46, ×6,38,
  ×6,25) ; les résultats, eux, ne varient pas.

## Volontairement non fait (et ce qui le ferait revenir)

| Non fait | Pourquoi pas maintenant | Déclencheur |
|---|---|---|
| Thread de décodage, NVDEC, TensorRT, batching | ×6,5 le temps réel ; un GPU occupé la moitié du temps est acceptable pour une vidéo | plusieurs flux par GPU, ou coût à l'échelle |
| Tracking, affectation d'équipe, homographie du terrain | non demandés, pas de vérité terrain pour les évaluer | données annotées et besoin produit |
| Évaluation de la qualité du détecteur | pas de labels ; l'énoncé ne le demande pas | avant tout changement de modèle : un petit jeu annoté par type de plan |
| MLflow | pas d'entraînement ; le lignage est déjà dans l'en-tête et le résumé de sortie | plusieurs runs à comparer, ou un serveur de tracking existant |
| Build Docker et run GPU en CI | les runners hébergés n'ont pas de GPU ; la CI reconstruit `uv.lock` et lance les tests CPU | runner GPU auto-hébergé |

## Roadmap de production (non implémentée)

Entrée S3 → une exécution Step Functions par match → job GPU sur EKS en Spot (cette image, figée par digest ; version
du modèle résolue une seule fois) → sorties sur S3 sous une clé déterministe (`match/modèle/commit`). Le code de
sortie pilote l'orchestrateur (tableau ci-dessus) ; un match n'est traité que si `summary.status == "complete"`. Les
métriques de bloc alimentent Grafana (débit, p95, `speed`) ; détections par image, confiance et taux de ballon par
ligue et par diffuseur servent de signaux de dérive sans labels.

## Organisation

```text
src/vidinfer/video.py     métadonnées, décodage (+ compteurs d'anomalies d'entrée), échantillonnage au timestamp, conversion RGB + contrat
src/vidinfer/model.py     poids figés (URL + sha256), détecteur avec la frontière BGR
src/vidinfer/__main__.py  CLI, boucle, chronos par étape, logs logfmt, output.jsonl, statut et codes de sortie, empreintes
src/vidinfer/viz.py       frise parsée depuis run.log, images annotées contrôlées contre output.jsonl
tests/                    échantillonnage, pixels/RGB/matrice, frontière du modèle et poids, bout en bout, codes de sortie
results/                  run de référence + reproducibility.log (empreintes des 3 runs)
```
