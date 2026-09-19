# CNN "from scratch" vs Transfert Learning — Cats vs Dogs

Comparaison d'un **CNN entraîné from scratch** (Expérience A) et d'un **ResNet18 pré-entraîné ImageNet** adapté par transfert d'apprentissage (Expérience B) sur la classification chats / chiens. Tout le code est dans [`notebook.ipynb`](notebook.ipynb).

## Environnement

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

**GPU** : l'entraînement complet a été réalisé sur **Google Colab (GPU NVIDIA T4)**. Le notebook affiche le device utilisé au démarrage (`torch.cuda.is_available()`) et fonctionne aussi sur CPU, mais beaucoup plus lentement.

## Organisation des données

Le dataset n'est **pas versionné**. Téléchargez [Cat_Dog_data.zip](https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip) (issu de [Dogs vs Cats, Kaggle](https://www.kaggle.com/c/dogs-vs-cats)) et placez-le à côté du notebook :

```
cnn-catsdogs-OusmaneColy/
├─ notebook.ipynb
├─ Cat_Dog_data/
│  ├─ train/{cat,dog}/*.jpg     (11 250 images par classe)
│  └─ test/{cat,dog}/*.jpg      (1 250 images par classe)
├─ results/                     historiques et tableaux (versionnés)
└─ checkpoints/                 modèles .pt (non versionnés)
```

**Sur Colab** : placez `notebook.ipynb`, `Cat_Dog_data/` et `results/` dans le dossier `MyDrive/deep_learning` du Drive. Le notebook monte le Drive et s'y place (`os.chdir`). Lire les images directement depuis le Drive est lent à la 1re époque (~10 000 s observées) ; copier `Cat_Dog_data` sur le disque de Colab (`/content`) avant d'entraîner accélère beaucoup.

## Entraîner

Tout se lance depuis le notebook (`Exécuter tout`). Paramètres dans la cellule *Configuration* :

| Paramètre | Valeur |
|---|---|
| Taille d'image / batch | 224 / 32 |
| Époques | A : 15, B : 5 (le transfert converge dès les premières époques) |
| Optimiseurs testés | Adam (`lr=1e-3`), SGD (`lr=1e-2`, momentum 0.9) |
| Scheduler | `StepLR(step_size=5, gamma=0.5)` |
| Seed | 42 (Python, NumPy, PyTorch, cuDNN déterministe) |
| Split | 80 % train / 20 % validation (depuis `train/`), test à part |

| | A — From scratch | B — Transfert learning |
|---|---|---|
| Modèle | 4 blocs `Conv → BatchNorm → ReLU → MaxPool`, tête `Dropout(0.5) → Linear → Dropout(0.3) → Linear` | `resnet18` ImageNet, backbone gelé, tête `Dropout(0.5) → Linear(512, 2)` |
| BatchNorm | après chaque convolution (stabilise et accélère l'entraînement) | déjà présent dans le backbone pré-entraîné |
| Dropout | dans la tête dense, où se concentre le risque de sur-apprentissage | devant la nouvelle couche de classification |

Augmentation (train uniquement) : rotation ±20°, recadrage aléatoire, flip horizontal. Normalisation ImageNet.

**Reprise automatique** : chaque run enregistre à chaque époque son meilleur modèle (`checkpoints/<run>_best.pt`), un point de reprise (`<run>_last.pt`) et son historique (`results/<run>_history.json`). Relancer le notebook saute les runs terminés et reprend un run interrompu (utile quand la session Colab est coupée). Pour tout réentraîner : vider `results/` et `checkpoints/`, ou mettre `RESUME = False`.

**Vérification rapide sans GPU** : `FAST_DEBUG=1 jupyter nbconvert --to notebook --execute notebook.ipynb` exécute un mini-run (sous-échantillon, 2 époques) dont les fichiers vont dans `checkpoints/debug/` et `results/debug/`.

## Évaluer / recharger un modèle

La section *Test final* du notebook recrée des modèles neufs, recharge `checkpoints/<run>_best.pt` avec `load_state_dict`, évalue sur le jeu de test (loss, accuracy, précision, recall, matrice de confusion) et affiche des exemples d'erreurs.

## Résultats

### Validation (meilleure époque de chaque run, GPU T4)

| Run | Meilleure époque | Val loss | Val accuracy | Val précision | Val recall |
|---|---|---|---|---|---|
| A — from scratch, Adam | 13 / 15 | 0.346 | 0.854 | 0.845 | 0.866 |
| A — from scratch, SGD | 14 / 15 | 0.401 | 0.824 | 0.835 | 0.805 |
| B — transfert, Adam | 3 / 5 | 0.070 | 0.975 | 0.981 | 0.969 |
| B — transfert, SGD | 4 / 5 | 0.111 | 0.969 | 0.971 | 0.967 |

Temps par époque sur T4 : ~205-212 s (A) et ~179 s (B-Adam) avec lecture depuis le Drive ; ~115 s (B-SGD) avec les images copiées sur le disque de Colab.

> Les historiques des runs A (Adam, SGD) et B (Adam) ont été reconstitués depuis les logs de la 1re session Colab, interrompue faute de ressources ; leurs modèles (`*_best.pt`) proviennent de cette session. B-SGD et le test final ont été exécutés dans la session de reprise. Les fichiers `results/B_transfer_sgd_history.json`, `validation_summary.csv` et `test_summary.csv` ont été recopiés depuis les sorties du notebook exécuté.

### Test final (modèles rechargés depuis `checkpoints/`, 2 516 images de test)

| Modèle | Test loss | Accuracy | Précision | Recall |
|---|---|---|---|---|
| A — from scratch (Adam) | 0.364 | 0.841 | 0.850 | 0.829 |
| B — transfert learning (Adam) | 0.069 | 0.974 | 0.982 | 0.967 |

Le test confirme la validation (0.854 → 0.841 pour A, 0.975 → 0.974 pour B) : pas de sur-ajustement au jeu de validation. Les matrices de confusion et des exemples d'erreurs sont dans `notebook.ipynb`.

### Analyse

- **Convergence** : le transfert learning atteint ~97 % de précision de validation dès la 1re époque ; le CNN from scratch part à ~70 % et n'atteint ~85 % qu'après 13 à 15 époques. Les features ImageNet sont directement exploitables, avec seulement la tête à entraîner.
- **Optimiseurs** : pour le modèle from scratch, Adam converge un peu plus vite que SGD (val accuracy 0.854 contre 0.824). Les deux restent bruités en validation, SGD davantage (recall entre 0.60 et 0.96 selon les époques).
- **Transfert learning, Adam vs SGD** : les deux atteignent ~97 % dès la 1re époque (Adam 0.975, SGD 0.969) ; SGD est plus irrégulier (val accuracy 0.947 à l'époque 3, avec un recall de 0.897).
- **Gain du transfert learning** : +13 points d'accuracy sur le test (0.974 contre 0.841), avec une loss 5 fois plus faible, pour un temps d'entraînement comparable par époque et 3 fois moins d'époques.
- **Régularisation et robustesse** : l'accuracy d'entraînement est inférieure à celle de validation (Dropout et augmentation actifs seulement à l'entraînement) : pas de sur-apprentissage visible. Le modèle de transfert est stable d'une époque à l'autre, le CNN from scratch beaucoup moins.

## Limites et pistes

- Backbone entièrement gelé : un fine-tuning des derniers blocs de ResNet18 pourrait encore progresser.
- Pas de recherche d'hyperparamètres systématique (seulement Adam vs SGD et un `StepLR`) ; un seul backbone testé (MobileNet, EfficientNet possibles).
- Pas de journalisation TensorBoard / W&B : le suivi passe par les courbes du notebook et les fichiers `results/`.
- Session Colab interrompue : B-Adam limité à 5 époques (suffisant, la validation plafonne dès l'époque 3).

## Licence

MIT, voir [LICENSE](LICENSE).
