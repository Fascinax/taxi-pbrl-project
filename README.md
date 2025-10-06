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

**1. Entraînement de l'agent classique:**
```bash
python train_classical_agent.py
```

**2. Démonstration du système de préférences:**
```bash
python demo_preferences.py
```

**3. Entraînement et comparaison avec l'agent PbRL:**
```bash
python train_pbrl_agent.py
```

### Modes d'entraînement PbRL:
1. **Mode automatique**: Utilise des préférences simulées
2. **Mode interactif**: Collecte tes préférences en temps réel
3. **Mode standard**: Agent normal pour comparaison

### Fichiers disponibles

**Agents:**
- `src/q_learning_agent.py`: Agent Q-Learning classique
- `src/pbrl_agent.py`: Agent Preference-based RL
- `src/trajectory_manager.py`: Gestion et comparaison des trajectoires
- `src/preference_interface.py`: Interface de collecte de préférences

**Scripts principaux:**
- `train_classical_agent.py`: Entraînement agent classique
- `train_pbrl_agent.py`: Entraînement agent PbRL et comparaison
- `demo_preferences.py`: Démonstration du système de préférences

**Tests:**
- `notebooks/test_taxi_env.py`: Test de base de l'environnement

## Phases terminées ✅

### Phase 1: Agent Q-Learning Classique ✅
- ✅ Agent Q-Learning fonctionnel
- ✅ Entraînement et évaluation
- ✅ Sauvegarde et métriques

### Phase 2: Système de Préférences ✅
- ✅ Module de comparaison de trajectoires
- ✅ Interface de saisie des préférences
- ✅ Visualisation et analyse automatique

### Phase 3: Agent PbRL (Preference-based RL) ✅
- ✅ Agent PbRL avec apprentissage par préférences
- ✅ Boucle d'apprentissage interactive
- ✅ Conversion préférences → signal d'apprentissage

## Prochaines étapes

### Phase 4: Expérimentations 🚀
- [ ] Comparaison complète classique vs PbRL
- [ ] Tests avec différents types de préférences
- [ ] Métriques et analyses détaillées
- [ ] Rédaction du rapport final

## Environnement Taxi-v3

- **États**: 500 (position taxi, passager, destination)
- **Actions**: 6 (Nord, Sud, Est, Ouest, Prendre, Déposer)
- **Objectif**: Prendre le passager et le déposer à destination
- **Récompenses**: 
  - +20: livraison réussie
  - -10: action illégale (prendre/déposer)
  - -1: chaque pas de temps