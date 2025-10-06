
# 📈 RAPPORT DE PERFORMANCE STATISTIQUE

## Résumé Exécutif
- **Performance PbRL**: 8.11 ± 2.42
- **Performance Classique**: 7.95 ± 2.70
- **Amélioration**: 2.01%
- **Significativité statistique**: ❌ Non (α = 0.05)

## Tests Statistiques

### Normalité des Distributions
- **Agent Classique**: Shapiro-Wilk p = 0.0172
- **Agent PbRL**: Shapiro-Wilk p = 0.0387
- **Conclusion**: Distributions non-normales

### Tests de Comparaison
1. **Test t de Student**: t = 0.442, p = 0.3296
2. **Mann-Whitney U**: U = 5214, p = 0.2993
3. **Kolmogorov-Smirnov**: D = 0.080, p = 0.9084

### Taille d'Effet
- **Cohen's d**: 0.062 (Effet négligeable)

### Intervalle de Confiance (95%)
- **Différence moyenne**: 0.160
- **IC 95%**: [-0.554, 0.874]

## Interprétation

### Significativité Pratique
L'amélioration du PbRL n'est pas statistiquement significative.

### Recommandations
- Collecter plus de données pour confirmer les tendances
- L'effet est trop petit pour être pratiquement significatif
- Efficacité d'entraînement: 2000 vs 15000 épisodes

---
*Rapport généré automatiquement - 2025-10-06 15:38:40*
