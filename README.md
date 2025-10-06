# Preference-based RL on Taxi-v3 Project

## Structure du projet

```
taxi-pbrl-project/
├── notebooks/          # Notebooks Jupyter pour expérimentation
│   └── test_taxi_env.py
├── src/               # Code source principal
│   └── q_learning_agent.py
├── results/           # Résultats et modèles sauvegardés
├── docs/             # Documentation
└── train_classical_agent.py  # Script d'entraînement
```

## Installation

1. Cloner/créer le projet
2. Installer les dépendances:
   ```bash
   pip install gymnasium numpy matplotlib pygame
   pip install "gymnasium[toy-text]"
   ```

## Phase 1: Agent Q-Learning Classique ✅

### Utilisation

```bash
python train_classical_agent.py
```

Cela va:
- Entraîner un agent Q-Learning sur Taxi-v3 (15000 épisodes)
- Évaluer ses performances
- Sauvegarder l'agent et les résultats
- Générer des graphiques de progression

### Fichiers créés
- `q_learning_agent.py`: Classe agent Q-Learning
- `train_classical_agent.py`: Script d'entraînement principal

## Prochaines étapes

### Phase 2: Système de Préférences
- [ ] Module de comparaison de trajectoires
- [ ] Interface de saisie des préférences
- [ ] Conversion préférences → signal d'apprentissage

### Phase 3: Agent PbRL (Preference-based RL)
- [ ] Modification de l'agent pour apprendre des préférences
- [ ] Boucle d'apprentissage interactive

### Phase 4: Expérimentations
- [ ] Comparaison classique vs PbRL
- [ ] Tests avec différents types de préférences
- [ ] Métriques et analyses

## Environnement Taxi-v3

- **États**: 500 (position taxi, passager, destination)
- **Actions**: 6 (Nord, Sud, Est, Ouest, Prendre, Déposer)
- **Objectif**: Prendre le passager et le déposer à destination
- **Récompenses**: 
  - +20: livraison réussie
  - -10: action illégale (prendre/déposer)
  - -1: chaque pas de temps