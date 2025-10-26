# 🎬 Guide Rapide - Visualisation Graphique des Préférences

## 🚀 Démarrage Rapide

### Option 1 : Test Rapide (Recommandé)

La façon la plus simple de tester la nouvelle visualisation :

```bash
python test_visual_preference.py
```

**Ce qui se passe** :
1. 🤖 Charge l'agent entraîné (ou en entraîne un rapidement)
2. 🎬 Collecte 2 trajectoires
3. 📺 Lance la visualisation graphique automatiquement
4. ⌨️ Vous demande votre préférence

---

### Option 2 : Démonstration Complète

Pour une session complète avec plusieurs comparaisons :

```bash
python demo_preferences.py
```

**Ce qui change** :
- ✨ La visualisation Gymnasium s'affiche **automatiquement** avant chaque choix
- 🎯 Plus besoin de taper 'viz' - c'est le comportement par défaut
- 📊 Le mode texte reste disponible avec la commande `text`

---

## 🎮 Contrôles Pendant la Visualisation

### Dans la Fenêtre Gymnasium

| Touche | Action |
|--------|--------|
| `ESPACE` | ⏯️ Pause / Reprise |
| `ÉCHAP` | ⏭️ Passer au choix |
| `Fermer fenêtre` | ⏭️ Passer au choix |

### Dans le Terminal (Après la Visualisation)

| Commande | Description |
|----------|-------------|
| `1` | 🔵 Je préfère la Trajectoire A (cyan) |
| `2` | 🟠 Je préfère la Trajectoire B (orange) |
| `0` | ⚖️ Les deux sont équivalentes |
| `replay` | 🔄 Rejouer la visualisation graphique |
| `text` | 📊 Afficher la comparaison textuelle détaillée |
| `help` | 🆘 Afficher l'aide |

---

## 💡 Comprendre l'Affichage

### Pendant le Replay

```
┌─────────────────────────────────────────────────────────┐
│              TRAJECTOIRE A          TRAJECTOIRE B       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Image Taxi]  │  [Image Taxi]                         │
│                                                         │
│  Pas: 5/14     │  Pas: 5/7                             │
│  Action: ↑     │  Action: ↓                            │
│  Récompense: -1│  Récompense: -1                       │
│  Cumulée: -5   │  Cumulée: -5                          │
└─────────────────────────────────────────────────────────┘
```

**Légende des Actions** :
- ↑ = Nord
- ↓ = Sud  
- → = Est
- ← = Ouest
- 🚖 = Prendre le passager
- 🎯 = Déposer le passager

**Codes Couleur** :
- 🔵 **Cyan/Bleu** = Trajectoire A (la première)
- 🟠 **Orange** = Trajectoire B (la seconde)
- 🟢 **Vert** = Trajectoire terminée

---

## 📊 Critères de Décision

Quand vous comparez deux trajectoires, considérez :

### 1. **Récompense Totale** ⭐
- Plus élevée = meilleure (généralement)
- Récompense positive = succès
- Ex: +13 est mieux que +7

### 2. **Efficacité** 🎯
- = Récompense totale / Nombre de pas
- Une trajectoire courte et réussie est très efficace
- Ex: +14 en 7 pas (2.0) > +7 en 14 pas (0.5)

### 3. **Longueur** 📏
- Plus court n'est pas toujours mieux !
- Une trajectoire courte qui échoue < trajectoire longue qui réussit
- Mais : trajectoire courte qui réussit > trajectoire longue qui réussit

### 4. **Style de Navigation** 🗺️
- Certaines stratégies sont plus "directes"
- D'autres font des détours
- À vous de juger ce qui vous semble "mieux"

---

## 🔧 Configuration

### Désactiver la Visualisation Graphique

Si vous préférez le mode texte uniquement :

```python
# Dans votre code
choice = preference_interface.collect_preference_interactive(
    traj1, traj2, traj_manager, 
    use_visual=False  # Force le mode texte
)
```

### Ajuster la Vitesse de Replay

```python
from src.visual_trajectory_comparator import VisualTrajectoryComparator

visualizer = VisualTrajectoryComparator()
visualizer.replay_trajectories_side_by_side(
    traj1, traj2, 
    delay=0.5  # Plus lent (défaut: 0.3)
)
visualizer.close()
```

**Valeurs suggérées** :
- `0.1` : Très rapide (difficile à suivre)
- `0.3` : **Normal** (recommandé)
- `0.5` : Lent (pour analyser en détail)
- `1.0` : Très lent (pour débutants)

---

## 🎓 Exemple de Session

### Scénario Typique

1. **Lancement** :
   ```bash
   python demo_preferences.py
   ```

2. **Visualisation automatique** :
   - 🎬 La fenêtre Gymnasium s'ouvre
   - 🚕 Les deux taxis se déplacent en parallèle
   - 📊 Les statistiques s'affichent en temps réel

3. **Observation** :
   - Vous voyez que la Trajectoire B est beaucoup plus directe
   - Elle prend le passager en 2 pas, la A en 6 pas
   - B livre le passager plus vite

4. **Décision** :
   ```
   👉 Votre choix: 2  (Je préfère la B)
   💭 Pourquoi: Plus directe et efficace
   ```

5. **Apprentissage** :
   - 🧠 L'agent apprend que vous préférez les trajectoires directes
   - 🔄 Il ajuste sa stratégie en conséquence

---

## ❓ FAQ

### Q: La fenêtre ne s'affiche pas ?
**R:** Vérifiez que pygame est installé :
```bash
pip install pygame
```

### Q: Je veux revoir la visualisation ?
**R:** Tapez `replay` quand le terminal demande votre choix.

### Q: Je préfère le mode texte ?
**R:** Tapez `text` pour voir les détails textuels, ou ajoutez `use_visual=False` dans le code.

### Q: Comment accélérer/ralentir ?
**R:** Utilisez le paramètre `delay` (voir section Configuration ci-dessus).

### Q: Puis-je enregistrer une vidéo ?
**R:** Pas encore implémenté, mais c'est dans les améliorations futures prévues !

---

## 🎉 Profitez de votre Nouvelle Expérience !

La visualisation graphique rend l'évaluation des trajectoires :
- ✅ Plus intuitive
- ✅ Plus rapide
- ✅ Plus engageante
- ✅ Plus précise

**Bon apprentissage par préférences !** 🚕🎯
