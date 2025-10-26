# 🎬 Visualisation Graphique des Préférences - Changelog

## 📅 Date: 22 octobre 2025

## ✨ Nouvelle Fonctionnalité Majeure

### Visualisation Gymnasium Interactive des Trajectoires

Avant cette mise à jour, la comparaison de trajectoires se faisait uniquement via du texte dans le terminal. Maintenant, vous pouvez **voir visuellement** les deux trajectoires se dérouler côte à côte avec le rendu Gymnasium officiel !

---

## 🎯 Ce qui a changé

### 1. Nouveau Module: `visual_trajectory_comparator.py`

Un nouveau module complet pour la visualisation graphique des trajectoires :

- **Affichage côte à côte** : Deux environnements Taxi-v3 sont affichés simultanément
- **Replay synchronisé** : Les trajectoires se déroulent en parallèle, pas par pas
- **Informations en temps réel** :
  - Numéro du pas actuel
  - Action effectuée (avec emojis : ↑↓←→🚖🎯)
  - Récompense instantanée
  - Récompense cumulée
- **Résumé final** : Tableau comparatif complet à la fin de la visualisation
- **Contrôles interactifs** :
  - `ESPACE` : Pause/Play
  - `ÉCHAP` : Passer à la saisie du choix
  - Fermeture de fenêtre : Continuer

**Couleurs distinctives** :
- 🔵 **Trajectoire A** : Cyan/Bleu
- 🟠 **Trajectoire B** : Orange

### 2. Interface de Préférences Améliorée

`preference_interface.py` a été mis à jour pour intégrer la visualisation :

**Nouveau paramètre** : `use_visual=True` (par défaut)
- Active automatiquement la visualisation Gymnasium
- Fallback intelligent vers le mode texte en cas d'erreur

**Nouvelles commandes utilisateur** :
- `replay` : Rejouer la visualisation graphique
- `text` : Afficher la comparaison textuelle détaillée
- `1` / `2` / `0` : Faire son choix (inchangé)
- `help` : Aide (inchangé)

**Ancien `viz`** : Toujours disponible pour rétrocompatibilité (affiche les graphiques matplotlib)

---

## 🚀 Utilisation

### Usage Simple

```python
from src.preference_interface import PreferenceInterface
from src.trajectory_manager import TrajectoryManager

# Collecter des trajectoires
traj_manager = TrajectoryManager()
traj1 = traj_manager.collect_trajectory(env, agent)
traj2 = traj_manager.collect_trajectory(env, agent)

# Interface avec visualisation graphique automatique
preference_interface = PreferenceInterface()
choice = preference_interface.collect_preference_interactive(
    traj1, traj2, traj_manager, use_visual=True  # Par défaut
)
```

### Désactiver la Visualisation (Mode Texte Uniquement)

```python
choice = preference_interface.collect_preference_interactive(
    traj1, traj2, traj_manager, use_visual=False
)
```

### Script de Test Dédié

Un nouveau script `test_visual_preference.py` a été créé pour tester rapidement la fonctionnalité :

```bash
python test_visual_preference.py
```

---

## 🎨 Captures d'Écran Conceptuelles

**Pendant le Replay** :
```
┌─────────────────────────────────┬─────────────────────────────────┐
│   TRAJECTOIRE A (Cyan)          │   TRAJECTOIRE B (Orange)        │
├─────────────────────────────────┼─────────────────────────────────┤
│   [Rendu Taxi-v3]               │   [Rendu Taxi-v3]               │
│                                 │                                 │
│   Pas: 5/14                     │   Pas: 5/7                      │
│   Action: Nord ↑                │   Action: Sud ↓                 │
│   Récompense: -1.0              │   Récompense: -1.0              │
│   Cumulée: -5.0                 │   Cumulée: -5.0                 │
└─────────────────────────────────┴─────────────────────────────────┘
         ESPACE: Pause/Play  |  ÉCHAP: Passer au choix
```

**Résumé Final** :
```
════════════════════════════════════════════════════════════════════
                  RÉSUMÉ DE LA COMPARAISON
════════════════════════════════════════════════════════════════════

Métrique            Trajectoire A      Trajectoire B      Meilleure
────────────────────────────────────────────────────────────────────
Récompense totale   +6.0               +13.0              B
Longueur            15                 8                  B
Efficacité          0.400              1.625              B
Succès              ✓                  ✓                  -
════════════════════════════════════════════════════════════════════
```

---

## 🔧 Détails Techniques

### Dépendances
- `pygame` : Pour le rendu de la fenêtre et les contrôles
- `gymnasium` : Rendu RGB natif de Taxi-v3
- Aucune dépendance supplémentaire requise (déjà dans requirements.txt)

### Architecture
- **Modularité** : Le visualiseur est un module séparé, réutilisable
- **Robustesse** : Gestion d'erreurs avec fallback vers le mode texte
- **Performance** : Délai configurable entre les pas (défaut: 0.3s)

### Points Clés du Code
1. Deux environnements Gymnasium indépendants (rgb_array mode)
2. Manipulation directe de `env.unwrapped.s` pour rejouer les états exacts
3. Boucle d'événements pygame pour les contrôles interactifs
4. Synchronisation intelligente des trajectoires de longueurs différentes

---

## 📈 Avantages

✅ **Expérience utilisateur améliorée** : Voir visuellement > lire du texte
✅ **Meilleure compréhension** : Les différences entre trajectoires sont évidentes
✅ **Plus engageant** : Interface graphique moderne et interactive
✅ **Rétrocompatible** : Le mode texte reste disponible
✅ **Flexible** : Possibilité de rejouer autant de fois que souhaité

---

## 🐛 Corrections et Ajustements

### Problème résolu : `env.render()` avant `env.reset()`
**Erreur initiale** :
```
Cannot call `env.render()` before calling `env.reset()`
```

**Solution** : Appeler `env.reset()` immédiatement après la création des environnements.

---

## 📝 Fichiers Modifiés

1. ✨ **Nouveau** : `src/visual_trajectory_comparator.py` (~300 lignes)
2. ✨ **Nouveau** : `test_visual_preference.py` (~70 lignes)
3. 🔧 **Modifié** : `src/preference_interface.py`
   - Ajout du paramètre `use_visual`
   - Intégration du visualiseur
   - Nouvelles commandes (`replay`, `text`)
4. 📚 **Nouveau** : `CHANGELOG_VISUAL_REPLAYS.md` (ce document)

---

## 🎓 Exemples d'Utilisation

### Cas d'Usage 1 : Entraînement PbRL Interactif

```python
# Dans train_pbrl_agent.py ou demo_preferences.py
agent.interactive_training_loop(
    env, preference_interface, traj_manager,
    episodes_per_iteration=1000,
    max_iterations=5
)
# → Les préférences utilisent automatiquement la visualisation !
```

### Cas d'Usage 2 : Analyse Rapide de Deux Stratégies

```python
# Comparer deux agents différents visuellement
agent1 = QLearningAgent(...)  # Agent classique
agent2 = PreferenceBasedQLearning(...)  # Agent PbRL

traj1 = traj_manager.collect_trajectory(env, agent1)
traj2 = traj_manager.collect_trajectory(env, agent2)

visualizer = VisualTrajectoryComparator()
visualizer.replay_trajectories_side_by_side(traj1, traj2)
visualizer.close()
```

---

## 🔮 Améliorations Futures Possibles

- [ ] Enregistrement vidéo des replays (mp4)
- [ ] Vitesse de replay ajustable dynamiquement
- [ ] Mode "auto-play" pour batch de comparaisons
- [ ] Overlay des Q-values sur les états
- [ ] Heatmap des états visités

---

## ✅ Tests Effectués

- ✅ Visualisation de trajectoires de longueurs différentes
- ✅ Pause/Play pendant le replay
- ✅ Fermeture anticipée (ÉCHAP)
- ✅ Commande `replay` pour revoir
- ✅ Fallback vers mode texte en cas d'erreur pygame
- ✅ Compatibilité avec l'ancien code (rétrocompatibilité)

---

## 📞 Support

En cas de problème :
1. Vérifier que `pygame` est installé : `pip install pygame`
2. Essayer le mode texte : `use_visual=False`
3. Tester avec : `python test_visual_preference.py`

---

**Profitez de cette nouvelle expérience visuelle pour évaluer vos préférences ! 🎉**
