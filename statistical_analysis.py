import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import pandas as pd
from typing import Dict, Any, Tuple, List

def load_results() -> Dict[str, Any]:
    """Charge les résultats de la comparaison"""
    with open("results/detailed_comparison.json", 'r') as f:
        return json.load(f)

def statistical_analysis(classical_scores: List[float], pbrl_scores: List[float]) -> Dict[str, Any]:
    """
    Effectue une analyse statistique complète
    
    Args:
        classical_scores: Scores de l'agent classique
        pbrl_scores: Scores de l'agent PbRL
        
    Returns:
        Dictionnaire contenant tous les tests statistiques
    """
    
    # Tests de normalité
    classical_shapiro = stats.shapiro(classical_scores)
    pbrl_shapiro = stats.shapiro(pbrl_scores)
    
    # Test t de Student (si distributions normales)
    t_stat, t_pvalue = stats.ttest_ind(pbrl_scores, classical_scores, alternative='greater')
    
    # Test de Mann-Whitney U (non-paramétrique)
    u_stat, u_pvalue = stats.mannwhitneyu(pbrl_scores, classical_scores, alternative='greater')
    
    # Test de Kolmogorov-Smirnov (différence de distributions)
    ks_stat, ks_pvalue = stats.ks_2samp(classical_scores, pbrl_scores)
    
    # Effet size (Cohen's d)
    pooled_std = np.sqrt(((len(classical_scores) - 1) * np.var(classical_scores, ddof=1) + 
                         (len(pbrl_scores) - 1) * np.var(pbrl_scores, ddof=1)) / 
                        (len(classical_scores) + len(pbrl_scores) - 2))
    cohens_d = (np.mean(pbrl_scores) - np.mean(classical_scores)) / pooled_std
    
    # Intervalle de confiance pour la différence de moyennes
    n1, n2 = len(classical_scores), len(pbrl_scores)
    s1, s2 = np.std(classical_scores, ddof=1), np.std(pbrl_scores, ddof=1)
    se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
    diff_mean = np.mean(pbrl_scores) - np.mean(classical_scores)
    
    # Degrés de liberté pour Welch's t-test
    df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    t_critical = stats.t.ppf(0.975, df)
    ci_lower = diff_mean - t_critical * se_diff
    ci_upper = diff_mean + t_critical * se_diff
    
    return {
        'normality': {
            'classical_shapiro': {'statistic': classical_shapiro.statistic, 'pvalue': classical_shapiro.pvalue},
            'pbrl_shapiro': {'statistic': pbrl_shapiro.statistic, 'pvalue': pbrl_shapiro.pvalue}
        },
        'comparison_tests': {
            't_test': {'statistic': t_stat, 'pvalue': t_pvalue},
            'mann_whitney_u': {'statistic': u_stat, 'pvalue': u_pvalue},
            'kolmogorov_smirnov': {'statistic': ks_stat, 'pvalue': ks_pvalue}
        },
        'effect_size': {
            'cohens_d': cohens_d,
            'interpretation': interpret_cohens_d(cohens_d)
        },
        'confidence_interval': {
            'difference_mean': diff_mean,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'degrees_freedom': df
        }
    }

def interpret_cohens_d(d: float) -> str:
    """Interprète la taille d'effet Cohen's d"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Effet négligeable"
    elif abs_d < 0.5:
        return "Petit effet"
    elif abs_d < 0.8:
        return "Effet moyen"
    else:
        return "Grand effet"

def create_advanced_visualizations(results: Dict[str, Any], stats_results: Dict[str, Any]):
    """Crée des visualisations avancées"""
    
    classical_scores = results['classical_agent']['evaluation_scores']
    pbrl_scores = results['pbrl_agent']['evaluation_scores']
    
    # Configuration du style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Distributions avec tests de normalité
    ax1 = plt.subplot(2, 4, 1)
    plt.hist(classical_scores, bins=15, alpha=0.7, label='Classique', density=True, color='skyblue')
    plt.hist(pbrl_scores, bins=15, alpha=0.7, label='PbRL', density=True, color='lightcoral')
    
    # Ajout des distributions normales théoriques
    x = np.linspace(min(min(classical_scores), min(pbrl_scores)), 
                    max(max(classical_scores), max(pbrl_scores)), 100)
    classical_normal = stats.norm.pdf(x, np.mean(classical_scores), np.std(classical_scores))
    pbrl_normal = stats.norm.pdf(x, np.mean(pbrl_scores), np.std(pbrl_scores))
    
    plt.plot(x, classical_normal, '--', color='blue', alpha=0.8, label='Normal théorique (C)')
    plt.plot(x, pbrl_normal, '--', color='red', alpha=0.8, label='Normal théorique (P)')
    
    plt.title('Distributions des Performances\n' + 
              f'Shapiro-Wilk p-values: C={stats_results["normality"]["classical_shapiro"]["pvalue"]:.3f}, ' +
              f'P={stats_results["normality"]["pbrl_shapiro"]["pvalue"]:.3f}')
    plt.xlabel('Récompense')
    plt.ylabel('Densité')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Q-Q plots pour normalité
    ax2 = plt.subplot(2, 4, 2)
    stats.probplot(classical_scores, dist="norm", plot=plt)
    plt.title('Q-Q Plot - Agent Classique')
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 4, 3)
    stats.probplot(pbrl_scores, dist="norm", plot=plt)
    plt.title('Q-Q Plot - Agent PbRL')
    plt.grid(True, alpha=0.3)
    
    # 3. Violin plot comparatif
    ax4 = plt.subplot(2, 4, 4)
    data_violin = [classical_scores, pbrl_scores]
    parts = plt.violinplot(data_violin, positions=[1, 2], showmeans=True, showmedians=True)
    plt.xticks([1, 2], ['Classique', 'PbRL'])
    plt.title('Distribution des Performances\n(Violin Plot)')
    plt.ylabel('Récompense')
    plt.grid(True, alpha=0.3)
    
    # Couleurs pour violin plot
    colors = ['skyblue', 'lightcoral']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # 4. Box plot avec outliers détaillés
    ax5 = plt.subplot(2, 4, 5)
    box_data = [classical_scores, pbrl_scores]
    bp = plt.boxplot(box_data, labels=['Classique', 'PbRL'], patch_artist=True)
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Ajout des moyennes
    means = [np.mean(classical_scores), np.mean(pbrl_scores)]
    plt.plot([1, 2], means, 'ro-', linewidth=2, markersize=8, label='Moyennes')
    
    plt.title('Box Plot Comparatif')
    plt.ylabel('Récompense')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Évolution des performances (si données temporelles disponibles)
    ax6 = plt.subplot(2, 4, 6)
    # Simulation d'une évolution temporelle basée sur l'ordre des scores
    plt.plot(range(len(classical_scores)), sorted(classical_scores, reverse=True), 
             'o-', alpha=0.7, label='Classique (triée)', color='blue')
    plt.plot(range(len(pbrl_scores)), sorted(pbrl_scores, reverse=True), 
             's-', alpha=0.7, label='PbRL (triée)', color='red')
    plt.title('Courbes de Performance\n(Scores triés par ordre décroissant)')
    plt.xlabel('Rang')
    plt.ylabel('Récompense')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Analyse de la différence
    ax7 = plt.subplot(2, 4, 7)
    diff_mean = stats_results['confidence_interval']['difference_mean']
    ci_lower = stats_results['confidence_interval']['ci_95_lower']
    ci_upper = stats_results['confidence_interval']['ci_95_upper']
    
    plt.bar(['Différence\nPbRL - Classique'], [diff_mean], 
            yerr=[[diff_mean - ci_lower], [ci_upper - diff_mean]], 
            capsize=10, color='green', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title(f'Différence de Moyennes\nIC 95%: [{ci_lower:.3f}, {ci_upper:.3f}]')
    plt.ylabel('Différence de récompense')
    plt.grid(True, alpha=0.3)
    
    # 7. Résumé des tests statistiques
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Tableau des résultats
    test_results = [
        ['Test t de Student', f"p = {stats_results['comparison_tests']['t_test']['pvalue']:.4f}"],
        ['Mann-Whitney U', f"p = {stats_results['comparison_tests']['mann_whitney_u']['pvalue']:.4f}"],
        ['Kolmogorov-Smirnov', f"p = {stats_results['comparison_tests']['kolmogorov_smirnov']['pvalue']:.4f}"],
        ['', ''],
        ['Cohen\'s d', f"{stats_results['effect_size']['cohens_d']:.3f}"],
        ['Interprétation', stats_results['effect_size']['interpretation']],
        ['', ''],
        ['Amélioration', f"{results['comparison']['improvement_percentage']:.2f}%"],
        ['Significativité α=0.05', 'Oui' if stats_results['comparison_tests']['t_test']['pvalue'] < 0.05 else 'Non']
    ]
    
    table = ax8.table(cellText=test_results,
                     colLabels=['Test / Métrique', 'Résultat'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.4, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Mise en forme du tableau
    for i in range(len(test_results)):
        if test_results[i][0] == '':  # Lignes vides
            table[(i+1, 0)].set_facecolor('#f0f0f0')
            table[(i+1, 1)].set_facecolor('#f0f0f0')
        elif 'Significativité' in test_results[i][0]:
            if test_results[i][1] == 'Oui':
                table[(i+1, 1)].set_facecolor('lightgreen')
            else:
                table[(i+1, 1)].set_facecolor('lightcoral')
    
    plt.suptitle('Analyse Statistique Complète - Agent Classique vs Agent PbRL', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Sauvegarde
    plt.savefig('results/advanced_statistical_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Analyse statistique avancée sauvegardée: results/advanced_statistical_analysis.png")
    plt.show()

def generate_performance_report(results: Dict[str, Any], stats_results: Dict[str, Any]):
    """Génère un rapport de performance détaillé"""
    
    classical_scores = results['classical_agent']['evaluation_scores']
    pbrl_scores = results['pbrl_agent']['evaluation_scores']
    
    report = f"""
# 📈 RAPPORT DE PERFORMANCE STATISTIQUE

## Résumé Exécutif
- **Performance PbRL**: {np.mean(pbrl_scores):.2f} ± {np.std(pbrl_scores, ddof=1):.2f}
- **Performance Classique**: {np.mean(classical_scores):.2f} ± {np.std(classical_scores, ddof=1):.2f}
- **Amélioration**: {results['comparison']['improvement_percentage']:.2f}%
- **Significativité statistique**: {'✅ Oui' if stats_results['comparison_tests']['t_test']['pvalue'] < 0.05 else '❌ Non'} (α = 0.05)

## Tests Statistiques

### Normalité des Distributions
- **Agent Classique**: Shapiro-Wilk p = {stats_results['normality']['classical_shapiro']['pvalue']:.4f}
- **Agent PbRL**: Shapiro-Wilk p = {stats_results['normality']['pbrl_shapiro']['pvalue']:.4f}
- **Conclusion**: {'Distributions normales' if min(stats_results['normality']['classical_shapiro']['pvalue'], stats_results['normality']['pbrl_shapiro']['pvalue']) > 0.05 else 'Distributions non-normales'}

### Tests de Comparaison
1. **Test t de Student**: t = {stats_results['comparison_tests']['t_test']['statistic']:.3f}, p = {stats_results['comparison_tests']['t_test']['pvalue']:.4f}
2. **Mann-Whitney U**: U = {stats_results['comparison_tests']['mann_whitney_u']['statistic']:.0f}, p = {stats_results['comparison_tests']['mann_whitney_u']['pvalue']:.4f}
3. **Kolmogorov-Smirnov**: D = {stats_results['comparison_tests']['kolmogorov_smirnov']['statistic']:.3f}, p = {stats_results['comparison_tests']['kolmogorov_smirnov']['pvalue']:.4f}

### Taille d'Effet
- **Cohen's d**: {stats_results['effect_size']['cohens_d']:.3f} ({stats_results['effect_size']['interpretation']})

### Intervalle de Confiance (95%)
- **Différence moyenne**: {stats_results['confidence_interval']['difference_mean']:.3f}
- **IC 95%**: [{stats_results['confidence_interval']['ci_95_lower']:.3f}, {stats_results['confidence_interval']['ci_95_upper']:.3f}]

## Interprétation

### Significativité Pratique
{'Le PbRL montre une amélioration statistiquement significative.' if stats_results['comparison_tests']['t_test']['pvalue'] < 0.05 else 'L\'amélioration du PbRL n\'est pas statistiquement significative.'}

### Recommandations
{'- Continuer le développement du PbRL' if stats_results['comparison_tests']['t_test']['pvalue'] < 0.05 else '- Collecter plus de données pour confirmer les tendances'}
{'- L\'effet est pratiquement significatif' if abs(stats_results['effect_size']['cohens_d']) > 0.2 else '- L\'effet est trop petit pour être pratiquement significatif'}
- Efficacité d'entraînement: {results['pbrl_agent']['training_episodes']} vs {results['classical_agent']['training_episodes']} épisodes

---
*Rapport généré automatiquement - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open('results/performance_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📋 Rapport de performance sauvegardé: results/performance_report.md")

def main():
    """Fonction principale pour l'analyse statistique avancée"""
    
    print("🔬 ANALYSE STATISTIQUE AVANCÉE")
    print("=" * 50)
    
    # Chargement des données
    results = load_results()
    classical_scores = results['classical_agent']['evaluation_scores']
    pbrl_scores = results['pbrl_agent']['evaluation_scores']
    
    print(f"📊 Données chargées:")
    print(f"   - Agent classique: {len(classical_scores)} évaluations")
    print(f"   - Agent PbRL: {len(pbrl_scores)} évaluations")
    
    # Analyse statistique
    print("\n🧮 Calcul des tests statistiques...")
    stats_results = statistical_analysis(classical_scores, pbrl_scores)
    
    # Visualisations
    print("\n📈 Génération des visualisations avancées...")
    create_advanced_visualizations(results, stats_results)
    
    # Rapport
    print("\n📝 Génération du rapport de performance...")
    generate_performance_report(results, stats_results)
    
    # Résumé console
    print("\n" + "=" * 50)
    print("🎯 RÉSULTATS CLÉS")
    print("=" * 50)
    print(f"Amélioration PbRL: {results['comparison']['improvement_percentage']:.2f}%")
    print(f"Cohen's d: {stats_results['effect_size']['cohens_d']:.3f} ({stats_results['effect_size']['interpretation']})")
    print(f"Significativité (p < 0.05): {'✅ Oui' if stats_results['comparison_tests']['t_test']['pvalue'] < 0.05 else '❌ Non'}")
    print(f"IC 95% différence: [{stats_results['confidence_interval']['ci_95_lower']:.3f}, {stats_results['confidence_interval']['ci_95_upper']:.3f}]")
    
    print("\n✅ Analyse statistique complète terminée!")

if __name__ == "__main__":
    main()