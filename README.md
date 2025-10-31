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



## Démarrage Rapide

```powershell
# Démonstrations
python demo_preferences.py
python demo_mountaincar.py

# Workflow Taxi
python train_classical_agent.py
python train_pbrl_agent.py

# Workflow MountainCar
python train_mountaincar_classical.py
python collect_mountaincar_preferences_auto.py
python train_mountaincar_pbrl.py

# Comparaison
python compare_taxi_vs_mountaincar.py
```

Consultez `COMMANDES_RAPIDES.md` pour plus de détails.

## Structure du Projet

```
taxi-pbrl-project/
├── train_*.py                      # Scripts d'entraînement
├── demo_*.py                       # Scripts de démonstration
├── collect_*.py                    # Scripts de collecte de préférences
├── compare_*.py                    # Scripts de comparaison
├── src/                            # Code source (agents, visualisation)
├── notebooks/                      # Tests et notebooks
├── README.md                       # Ce fichier
├── COMMANDES_RAPIDES.md            # Guide des commandes
└── requirements.txt                # Dépendances
```

## Documentation

- `COMMANDES_RAPIDES.md` - Guide des commandes
- `requirements.txt` - Dépendances Python

## Installation

```powershell
pip install -r requirements.txt
```
