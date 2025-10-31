
# COMMANDES ESSENTIELLES - AIDE-MÉMOIRE

## Installation

```powershell
# Installer les dépendances
pip install gymnasium numpy matplotlib
```

## Utilisation Rapide

### Option 1 : Visualisation Comparative
```powershell
python compare_taxi_vs_mountaincar.py
# Génère : results/comparison_taxi_vs_mountaincar_pbrl.png
```

### Option 2 : Démonstration Taxi
```powershell
python demo_preferences.py
python train_pbrl_agent.py
```

### Option 3 : Workflow Complet
```powershell
# Taxi
python train_classical_agent.py
python train_pbrl_agent.py

# MountainCar
python train_mountaincar_classical.py
python collect_mountaincar_preferences_auto.py
python train_mountaincar_pbrl.py

# Comparaison finale
python compare_taxi_vs_mountaincar.py
```
