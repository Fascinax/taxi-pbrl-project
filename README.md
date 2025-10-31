# Preference-Based Reinforcement Learning (PBRL) Project

## Vue d'Ensemble

Projet de comparaison d'agents **PBRL** vs **Classical RL** sur deux environnements contrastés :
- **Taxi-v3** : Environnement discret avec récompenses denses
- **MountainCar-v0** : Environnement continu avec récompenses sparses

## Résultats Principaux

| Environnement | PBRL Épisodes | Classical Épisodes | Réduction | Performance |
|---------------|---------------|-------------------|-----------|-------------|
| **Taxi-v3** | 2,000 | 15,000 | **-87%** | 7.77 ± 2.59 |
| **MountainCar** | 6,000 | 10,000 | **-40%** | -165.19 ± 19.94 |

**Conclusion clé :** Le PBRL atteint des performances équivalentes avec **40% à 87% moins d'épisodes** !

## Installation Rapide

```powershell
# Installer les dépendances
pip install gymnasium numpy matplotlib pygame
```

**Documentation** : `docs/visual_comparison_complete.md`

## Démarrage Rapide

### Option 1 : Comparaison Visuelle (1 min)
```powershell
python compare_taxi_vs_mountaincar.py
```

### Option 2 : Démonstration Taxi avec Visualisation (2 min)
```powershell
python demo_preferences.py
python train_pbrl_agent.py
```

### Option 3 : MountainCar avec Visualisation
```powershell
python train_mountaincar_classical.py
python collect_mountaincar_preferences.py
python train_mountaincar_pbrl.py
```

### Option 4 : Tests Rapides Visualisation
```powershell
python test_visual_preference.py      # Test Taxi
python test_mountaincar_visual.py     # Test MountainCar
```

### Option 5 : Workflow Complet
Voir **`GUIDE_UTILISATION.md`** pour le guide détaillé.

## Structure du Projet

```
taxi-pbrl-project/
├── SCRIPTS PRINCIPAUX
│   ├── train_classical_agent.py          # Taxi: Agent classique
│   ├── train_pbrl_agent.py               # Taxi: Agent PBRL
│   ├── train_mountaincar_classical.py    # MC: Agent classique
│   ├── train_mountaincar_pbrl.py         # MC: Agent PBRL
│   ├── compare_taxi_vs_mountaincar.py    # Comparaison inter-env
│   └── cleanup_project.py                # Nettoyage projet
│
├── Source CODE SOURCE (src/)
│   ├── q_learning_agent.py               # Agent Q-Learning base
│   ├── pbrl_agent.py                     # Agent PBRL (Taxi)
│   ├── mountain_car_agent.py             # Agent Q-Learning MC
│   ├── mountain_car_pbrl_agent.py        # Agent PBRL MC
│   └── ... (7 fichiers)
│
├── [PLOT] RÉSULTATS (results/)
│   ├── comparison_taxi_vs_mountaincar_pbrl.png  # ⭐ Comparaison visuelle
│   ├── comparison_insights.txt                  # ⭐ Analyse détaillée
│   ├── detailed_comparison.json                 # Données Taxi
│   └── mountaincar_pbrl_comparison.json         # Données MC
│
└── 📚 DOCUMENTATION
    ├── README.md                         # Ce fichier
    ├── GUIDE_UTILISATION.md              # ⭐ Guide complet
    ├── QUICKSTART.md                     # Guide rapide
    └── MOUNTAINCAR_RESULTS_FINAL.md      # Résultats MC
```

## Doc Documentation

- **⭐ `GUIDE_UTILISATION.md`** - Guide complet d'utilisation (COMMENCER ICI)
- **🎬 `docs/visual_comparison_complete.md`** - Guide visualisation Taxi & MountainCar
- **`docs/visual_preferences_guide.md`** - Guide détaillé visualisation Taxi
- **`QUICKSTART.md`** - Démarrage rapide
- **`MOUNTAINCAR_RESULTS_FINAL.md`** - Analyse détaillée MountainCar
- **`results/comparison_insights.txt`** - Insights comparatifs
- **`CHANGELOG_VISUAL_REPLAYS.md`** - Changelog visualisation

## Commandes Essentielles

```powershell
# Taxi TAXI - Workflow complet (7 min)
python train_classical_agent.py
python train_pbrl_agent.py

# MountainCar MOUNTAINCAR - Workflow complet (21 min)
python train_mountaincar_classical.py
python collect_mountaincar_preferences_auto.py
python train_mountaincar_pbrl.py

# [PLOT] COMPARAISON - Visualisation finale
python compare_taxi_vs_mountaincar.py

### Principaux Graphiques

1. **`results/comparison_taxi_vs_mountaincar_pbrl.png`**
   - Comparaison complète des deux environnements
   - 6 graphiques : efficacité, performance, stabilité, succès, etc.
   - Tableau de synthèse

2. **`results/comparison_classical_vs_pbrl.png`** (Taxi)
   - Courbes d'apprentissage
   - Distributions des récompenses

3. **`results/comparison_mountaincar_classical_vs_pbrl.png`** (MC)
   - Courbes d'apprentissage
   - Distributions des récompenses

### Données Brutes

- **`results/detailed_comparison.json`** - Taxi (100 épisodes d'évaluation)
- **`results/mountaincar_pbrl_comparison.json`** - MC (200 épisodes)
- **`results/comparison_taxi_vs_mountaincar.json`** - Comparaison

## Debug Dépannage

### Erreur de module
```powershell
pip install gymnasium numpy matplotlib
```

### Graphiques non visibles
Les graphiques sont sauvegardés automatiquement dans `results/`. Ouvrez les fichiers `.png` manuellement.

### Entraînement trop lent
Réduisez `n_episodes` dans les scripts d'entraînement.

## Support Support

Pour plus d'informations, consultez :
- **`GUIDE_UTILISATION.md`** - Guide complet et détaillé
- **`results/comparison_insights.txt`** - Analyse comparative
