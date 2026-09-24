# vidinfer

🇬🇧 [English version](README.md)

Décoder une vidéo, lancer une inférence à **10 inférences par seconde de vidéo** sur des images **RGB 720×1280**, et
loguer **la performance toutes les 10 inférences**, avec des résultats **reproductibles de façon vérifiable**.

Modèle : détection des personnes et du ballon (poids officiels Ultralytics YOLO26s, classes COCO `person` et
`sports ball`).

## Résultats sur `cut.mp4` (run de référence, `results/reference/`)

| | |
|---|---|
| Entrée | H.264 1920×1080, 25 fps constants, 7 500 images, 300 s, espace couleur non déclaré |
| Inférences | **3 000 / 3 000 attendues**, 7 500 / 7 500 images décodées, 0 erreur de décodage |
| Débit | **58,2 inférences/s = ×5,8 le temps réel** (51,6 s pour 300 s de vidéo), bloc le plus lent ×3,6 |
| Latence par image | p50 16,6 ms · p95 19,2 ms · p99 20,7 ms (hors chargement du modèle 0,1 s et préchauffage 2,7 s) |
| Répartition du temps | inférence 72 % · prétraitement 23 % · empreinte 3 % · attente décodage 2 % · écriture 0,5 % |
| GPU | RTX 4050 Laptop, pic de 144 Mio alloués |
| Reproductibilité | run hôte, second run hôte et run Docker : `results_sha256` et `frames_crc32` **identiques** |

Visuels dans `results/reference/viz/` : une frise performance/détections reconstruite depuis `run.log`, et 4 images
annotées (plans larges à t=20 s et t=240 s ; l'incrustation du classement à t=90 s et un gros plan à t=170 s comme cas d'échec).

![frise](results/reference/viz/timeline.png)

## Démarrage rapide

```bash
uv sync --locked                                              # environnement exact depuis uv.lock
uv run python -m vidinfer data/cut.mp4 --output-dir outputs   # -> outputs/output.jsonl, outputs/run.log
uv run python -m vidinfer.viz outputs                         # -> outputs/viz/*.png|jpg
uv run pytest                                                 # 11 tests, CPU seul, ~5 s, vidéos synthétiques
```

Docker (GPU via le NVIDIA Container Toolkit ; poids intégrés à l'image et vérifiés au build) :

```bash
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t vidinfer:0.1.0 .
docker run --rm --gpus all -v "$PWD/data:/data:ro" -v "$PWD/outputs:/work/outputs" vidinfer:0.1.0 /data/cut.mp4
```

Options : `--fps 10`, `--model yolo26s.pt|yolo26m.pt|chemin.pt`, `--device auto|cuda|cpu` (repli CPU ≈ 2 inf/s),
`--max-frames N` (test rapide). Codes de sortie : `0` succès, `2` entrée invalide (ne pas relancer), `1` erreur
inattendue.

## Exigence → implémentation → preuve

| Exigence | Implémentation | Preuve |
|---|---|---|
| Programme Python prenant `cut.mp4` en argument | `python -m vidinfer VIDEO` | `tests/test_cli.py` (bout en bout sur une vidéo synthétique) |
| Redimensionner en 720 (h) × 1280 (l) | un seul appel swscale : YUV→RGB + redimensionnement `AREA` (réduction anti-aliasée) ; même ratio 16:9, aucune déformation | contrat vérifié sur chaque image (`check_rgb`) |
| Garantir le RGB | `rgb24` par construction + contrat ; HD non déclarée → matrice BT.709 (les bibliothèques appliquent BT.601 par défaut), signalée en WARNING | test sur une vidéo rouge pur : le canal 0 est le rouge |
| 10 inférences par seconde de vidéo | échantillonnage au timestamp : image la plus proche de `t0 + k/10`, égalité → image antérieure, fractions exactes | `3000/3000` dans le résumé ; `test_sampler.py` ; vidéo dont chaque image porte son numéro dans ses pixels |
| Loguer le temps toutes les 10 images | une ligne logfmt `event=block` toutes les 10 inférences, avec le détail par étape | 300 lignes de bloc dans `run.log`, relues par les tests et par `viz` |
| Temps d'exécution total | `event=timing` : temps mural du programme, préparation, chargement du modèle, préchauffage, traitement, temps CPU | `run.log` |
| Quelques métadonnées vidéo | `event=video` : conteneur, codec, profil, taille, fps, time_base, durée, images, débit, tags couleur, sha256 | `run.log` + en-tête de `output.jsonl` |
| `output.<format>` pour chaque image | `output.jsonl` : en-tête / une ligne par image inférée (même sans détection) / résumé | `results/reference/output.jsonl` (4,3 Mo) |
| Images ou visualisation | `python -m vidinfer.viz` : frise depuis `run.log` + images annotées, re-décodées et contrôlées contre `output.jsonl` | `results/reference/viz/` |

## Méthode

Chaque décision a suivi la même boucle : **lire l'énoncé comme une spécification → observer la donnée avant de coder →
lister ce qui peut être faux sans bruit → mesurer les options sur la vraie vidéo → décider par une règle explicite →
verrouiller par un test qui échoue si la logique casse → s'arrêter** (ce qui n'est ni demandé, ni testé, ni bon marché
part en roadmap, avec la condition qui le ferait revenir).

| Décision | Options étudiées | Choix | Pourquoi |
|---|---|---|---|
| 25 → 10 fps | 1 image sur N, timestamps flottants, filtre `fps` de ffmpeg, *floor*, plus proche | image la plus proche, égalité → antérieure, `Fraction`, grille calée sur le premier PTS | 25/10 = 2,5 : une cible sur deux tombe exactement entre deux images. Les flottants tranchent par bruit d'arrondi ; le `fps=10` par défaut de ffmpeg prend les images 1,3,6,8… (20 à 40 ms de retard). « La plus proche » borne l'erreur à une demi-image (20 ms). |
| Décodeur | OpenCV, pipe ffmpeg, PyAV, torchcodec, NVDEC | PyAV, `thread_type="AUTO"` | PTS exacts, RGB et matrice couleur maîtrisés, CPU le plus bas ; décoder toutes les images est de toute façon obligatoire (P-frames H.264) |
| Couleur | défauts des bibliothèques / explicite | BT.709 explicite pour la HD non déclarée + WARNING | un rouge pur encodé en BT.601 et décodé en BT.709 revient avec G=23 au lieu de 0 (vu dans les tests) |
| Frontière du modèle | RGB tel quel / vue BGR | vue BGR uniquement à l'appel Ultralytics | Ultralytics traite un tableau numpy comme du BGR et l'inverse (`engine/predictor.py`) ; passer du RGB est une erreur silencieuse |
| Modèle | YOLO26 n/s/m, imgsz 640/1280, FP16/32, RF-DETR | YOLO26s, imgsz 1280, FP16, conf 0,25 | le plus petit modèle pas mesurablement moins bon sur le ballon (8 à 16 px en 720p, d'où imgsz 1280) ; m est à un paramètre près |
| Boucle | threads producteur/consommateur, batching | boucle synchrone | déjà ×5,8 le temps réel ; le décodage tourne dans les threads FFmpeg (0,27 ms d'attente par image) |
| Sortie | JSON, CSV, Parquet, COCO, MOT | JSON Lines | streamable, lisible après un crash, garde les images sans détection ; la ligne de résumé marque la complétude |

## Reproductibilité

La reproductibilité est **vérifiée, pas supposée** : chaque run se termine par deux empreintes.

- `results_sha256` : empreinte de `(index d'image source, détections)` pour chaque échantillon, temps exclus.
- `frames_crc32` : CRC de tous les pixels envoyés au modèle (0,5 ms/image, chronométré comme une étape à part). Si
  les résultats divergent un jour, elle dit si l'écart vient du décodage/prétraitement ou du modèle.

Preuve : `results/reproducibility.log` (lignes start, environnement, temps et empreintes des 3 runs).

| Run | Environnement | results_sha256 | frames_crc32 |
|---|---|---|---|
| référence | venv hôte, Python 3.12.13, glibc 2.43 | `824865c8…a40a9c` | `0fb35010` |
| second run | même hôte, second run | `824865c8…a40a9c` | `0fb35010` |
| Docker | image Docker, Python 3.12.14, glibc 2.36 | `824865c8…a40a9c` | `0fb35010` |

Ce qui est figé, et où c'est enregistré :

| Couche | Figée par | Enregistrée dans |
|---|---|---|
| Code | commit git (`-dirty` si modifié) | `event=start`, en-tête de sortie |
| Dépendances Python | `uv.lock` (versions + hashes ; torch depuis l'index cu126) | `event=environment` (versions, CUDA, cuDNN, GPU, driver) |
| Image système | image de base par digest, uv par version | Dockerfile |
| Poids du modèle | URL de release exacte + sha256, vérifié **avant** chargement (un `.pt` est un pickle) ; Docker `ADD --checksum` | `event=model`, en-tête de sortie |
| Entrée | sha256 de la vidéo | `event=video`, en-tête de sortie |
| Paramètres | argv + paramètres effectifs du modèle | en-tête de sortie |
| Effets de bord à l'exécution | `YOLO_OFFLINE=1` (ni télémétrie ni vérification en ligne), `YOLO_AUTOINSTALL=0` (aucun pip install en cours de run) | — |
| Reconstruction sur machine vierge | CI : `uv sync --locked` + lint + tests à chaque push | `.github/workflows/ci.yml` |

Par défaut, Ultralytics télécharge les poids depuis la *dernière* release GitHub : d'où l'URL figée.
Lancer l'image Docker (et pas seulement la construire) a révélé deux bugs avant la livraison : un `ADD` distant crée le
fichier en mode 600, et `--chmod` s'appliquait aussi au dossier parent créé implicitement (non traversable par
l'utilisateur non-root).

## Observabilité de la performance

Chaque ligne de log est au format **logfmt** : lisible par un humain, parsable en une ligne de Python, sans dépendance.

```text
2026-09-24 15:07:47.709 INFO    event=block block=150 samples=1490-1499 video_t_s=149.90 wall_ms=165.52 ms_per_frame=16.55 inf_per_s=60.42 speed=6.04 decode_ms=2.52 preprocess_ms=34.47 fingerprint_ms=5.00 inference_ms=122.30 write_ms=0.96 other_ms=0.27 yolo_preprocess_ms=9.87 yolo_inference_ms=73.32 yolo_postprocess_ms=7.80 cpu_util=1.76 gpu_mem_mb=50.25 elapsed_s=25.60
```

- `speed` = secondes de vidéo traitées par seconde réelle (le SLO temps réel) ; un bloc avec `speed < 1` émet un WARNING.
- Les étapes somment à `wall_ms` ; `other_ms` est le reste non expliqué (surcoût de boucle) : rien n'est caché.
- Les étapes GPU sont chronométrées après `torch.cuda.synchronize()` : CUDA est asynchrone, sans cela on mesure le
  lancement des kernels.
- `yolo_*_ms` sont les temps internes d'Ultralytics, gardés à côté des nôtres volontairement.
- `cpu_util` (secondes CPU / secondes réelles) et `gpu_mem_mb` signalent les phases limitées par le CPU et les fuites
  mémoire.

```python
import shlex
blocks = [dict(t.split("=", 1) for t in shlex.split(line.split(" event=block ", 1)[1]))
          for line in open("outputs/run.log") if " event=block " in line]
```

Ce que les logs ont montré sur ce run :

1. Mesuré à notre frontière, l'appel au modèle coûte **12,4 ms/image**, alors qu'Ultralytics annonce **9,1 ms**
   (1,0 prétraitement + 7,3 réseau + 0,8 NMS) : 3,3 ms par appel sont invisibles dans les temps de la bibliothèque.
   Il faut mesurer à la frontière.
2. Le décodage ne coûte que **0,27 ms d'attente par image** : les threads FFmpeg décodent en parallèle
   (`cpu_util` ≈ 1,7 cœur).
3. La première cible d'optimisation serait le **prétraitement (3,9 ms/image, 23 %)**, un appel swscale mono-thread
   YUV→RGB + redimensionnement. Inutile à ×5,8 le temps réel.

## Format de sortie (`output.jsonl`)

```json
{"type":"header","schema_version":"1.0.0","program":{"git_commit":"bac08dc…"},"environment":{…},"video":{"sha256":"4072bd68…",…},"processing":{"target_fps":10,"yuv_matrix":"ITU709",…},"model":{"weights":"yolo26s.pt","sha256":"646f8bc3…",…},"coordinates":"bbox_xyxy in pixels of the 1280x720 frame"}
{"type":"frame","sample_index":2400,"source_frame_index":6000,"pts":3072000,"timestamp_s":240.0,"inference_ms":12.566,"detections":[{"bbox_xyxy":[628.0,537.0,641.0,550.0],"score":0.8081,"class_id":32,"class_name":"sports ball"},{"bbox_xyxy":[492.5,470.0,551.5,559.0],"score":0.8916,"class_id":0,"class_name":"person"},…]}
{"type":"summary","n_inferences":3000,"expected":3000,"processing_wall_s":51.561,"speed_x":5.82,"latency_ms":{…},"results_sha256":"824865c8…","frames_crc32":"0fb35010"}
```

Une ligne `summary` absente signifie que le run ne s'est pas terminé. Les lecteurs ignorent les clés inconnues.

## Limites

- **Licence** : Ultralytics est sous AGPL-3.0 ; un usage en production demande une licence Enterprise ou un détecteur
  Apache-2.0 (RF-DETR a été évalué comme plan B pendant l'étude). Le détecteur est isolé derrière une seule classe :
  seul l'adaptateur changerait.
- **Détecteur COCO générique** : 77 % des échantillons contiennent une détection `sports ball` à conf ≥ 0,25. Ce n'est
  pas un rappel : cela inclut des ballons de réserve au bord du terrain. Gros plans, ralentis et incrustations donnent
  peu de détections ou des détections fausses (voir `frame_t170.0s.jpg`) ; 13 images n'ont aucune détection (générique
  d'intro).
- **BT.709 est une hypothèse** pour ce flux HD non déclaré (la convention usuelle) ; la matrice est loguée et un flux
  déclaré est toujours respecté.
- Les temps viennent d'un GPU de portable et varient un peu d'un run à l'autre (×5,82 en référence, ×5,68 au second
  run et sous Docker) ; les résultats, eux, ne varient pas (empreintes identiques).

## Volontairement non fait (et ce qui le ferait revenir)

| Non fait | Pourquoi pas maintenant | Déclencheur |
|---|---|---|
| Threads / batching / NVDEC / TensorRT | ×5,8 le temps réel ; le décodage recouvre déjà l'inférence | GPU inactif > 30 %, multi-flux, coût à l'échelle |
| Tracking, affectation d'équipe, homographie du terrain | non demandés, pas de vérité terrain pour les évaluer | données annotées et besoin produit |
| MLflow | pas d'entraînement ici ; le lignage est déjà dans l'en-tête et le résumé de sortie | plusieurs runs à comparer, ou un serveur de tracking existant |
| Tests GPU en CI | les runners hébergés n'ont pas de GPU ; la CI lance le lint et les tests CPU depuis `uv.lock` | runner GPU auto-hébergé |
| Sortie Parquet | binaire, illisible après un crash | volumes de production |

## Roadmap de production (non implémentée)

Entrée S3 → une exécution Step Functions par match → job GPU sur EKS en Spot (cette image, figée par digest, version du
modèle résolue une seule fois) → sorties sur S3 sous une clé déterministe (`match/modèle/commit`). La ligne `summary`
rend les relances idempotentes : présente = on saute, absente = on relance. Le code de sortie 2 n'est jamais relancé.
Les métriques de bloc vont dans Prometheus/Grafana (débit, p95, `speed`) ; détections par image, confiance et taux de
ballon par ligue et par diffuseur servent de signaux de dérive sans labels.

## Organisation

```text
src/vidinfer/video.py     métadonnées, décodage, échantillonnage au timestamp, conversion RGB + contrat
src/vidinfer/model.py     poids figés (URL + sha256), détecteur avec la frontière BGR
src/vidinfer/__main__.py  CLI, boucle, chronos par étape, logs logfmt, output.jsonl, empreintes
src/vidinfer/viz.py       frise parsée depuis run.log, images annotées
tests/                    échantillonnage, pixels/RGB, frontière du modèle, bout en bout (vidéos synthétiques)
results/                  run de référence + reproducibility.log (empreintes des 3 runs)
```
