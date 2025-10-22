# 🚀 QUICKSTART - Démarrage en 5 Minutes

## Installation (1 min)

```powershell
pip install gymnasium numpy matplotlib
```

## Visualisation Rapide (1 min) ⭐

```powershell
python compare_taxi_vs_mountaincar.py
```

**Ouvrir :** `results/comparison_taxi_vs_mountaincar_pbrl.png`

## Résultats Attendus

```
╔══════════════════════════════════════════════════════╗
║  PBRL vs CLASSICAL RL                                ║
╠══════════════════════════════════════════════════════╣
║  🚕 Taxi:       -87% d'épisodes  (2k vs 15k)        ║
║  🏔️ MountainCar: -40% d'épisodes  (6k vs 10k)        ║
╚══════════════════════════════════════════════════════╝
```

## Documentation Complète

- **INDEX_DOCUMENTATION.md** → Guide de navigation
- **GUIDE_UTILISATION.md** → Utilisation détaillée
- **RECAPITULATIF_FINAL.md** → Conseils rapport

## Reproduire Tout (30 min optionnel)

```powershell
# Taxi
python train_classical_agent.py
python train_pbrl_agent.py

# MountainCar
python train_mountaincar_classical.py
python collect_mountaincar_preferences_auto.py
python train_mountaincar_pbrl.py

# Comparaison
python compare_taxi_vs_mountaincar.py
```

---

**C'est tout ! Vous avez tout ce qu'il faut pour votre projet.** 🎉
