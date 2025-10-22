# 🚀 COMMANDES ESSENTIELLES - AIDE-MÉMOIRE

## 📋 Installation

```powershell
# Installer les dépendances
pip install gymnasium numpy matplotlib
```

## 🎯 Utilisation Rapide

### Option 1 : Visualisation Comparative (1 min) ⭐
```powershell
python compare_taxi_vs_mountaincar.py
# Génère : results/comparison_taxi_vs_mountaincar_pbrl.png
```

### Option 2 : Démonstration Taxi (2 min)
```powershell
python demo_preferences.py
python train_pbrl_agent.py
```

### Option 3 : Workflow Complet (30 min)
```powershell
# Taxi (7 min)
python train_classical_agent.py
python train_pbrl_agent.py

# MountainCar (21 min)
python train_mountaincar_classical.py
python collect_mountaincar_preferences_auto.py
python train_mountaincar_pbrl.py

# Comparaison finale
python compare_taxi_vs_mountaincar.py
```

## 🧹 Nettoyage

```powershell
# Supprimer fichiers obsolètes (interactif)
python cleanup_project.py
```

## 📊 Fichiers Importants à Consulter

```powershell
# Visualisations
results/comparison_taxi_vs_mountaincar_pbrl.png  # ⭐ LE PLUS IMPORTANT
results/comparison_classical_vs_pbrl.png         # Taxi
results/comparison_mountaincar_classical_vs_pbrl.png  # MountainCar

# Analyses
results/comparison_insights.txt                  # Analyse textuelle
results/detailed_comparison.json                 # Données Taxi
results/mountaincar_pbrl_comparison.json         # Données MC

# Documentation
README.md                                        # Vue d'ensemble
GUIDE_UTILISATION.md                             # Guide complet ⭐
MOUNTAINCAR_RESULTS_FINAL.md                     # Analyse MC détaillée
RECAPITULATIF_FINAL.md                           # Ce que vous lisez
```

## 🎯 Résultats Attendus

### Taxi-v3
- **PBRL** : 2,000 épisodes → 7.77 ± 2.59
- **Classical** : 15,000 épisodes → 7.82 ± 2.60
- **Réduction** : **-87%** d'épisodes ✅

### MountainCar-v0
- **PBRL** : 6,000 épisodes → -165.19 ± 19.94 (77% succès)
- **Classical** : 10,000 épisodes → -153.53 ± 3.76 (100% succès)
- **Réduction** : **-40%** d'épisodes ✅

## 🐛 Dépannage Rapide

```powershell
# Erreur de module
pip install gymnasium numpy matplotlib

# Vérifier que vous êtes dans le bon dossier
cd taxi-pbrl-project
```

## 📞 Documentation

- **README.md** - Vue d'ensemble
- **GUIDE_UTILISATION.md** - Guide complet (⭐ COMMENCER ICI)
- **RECAPITULATIF_FINAL.md** - Résumé et conseils rapport

---

**Temps total** : 30 minutes (workflow complet)  
**Résultat** : PBRL réduit les épisodes de 40% à 87% ! 🚀
