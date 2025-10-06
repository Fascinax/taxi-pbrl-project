import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from src.pbrl_agent import PreferenceBasedQLearning
from src.q_learning_agent import QLearningAgent
from src.trajectory_manager import TrajectoryManager
from src.preference_interface import PreferenceInterface
import pickle
import os
import json

def main():
    """
    Script principal pour entraîner l'agent PbRL et le comparer avec l'agent classique
    """
    
    print("=== ENTRAÎNEMENT AGENT PREFERENCE-BASED RL ===")
    
    # Configuration
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Création de l'environnement
    env = gym.make("Taxi-v3")
    n_states = 500
    n_actions = 6
    
    # Initialisation des managers
    trajectory_manager = TrajectoryManager()
    preference_interface = PreferenceInterface()
    
    print("\n1️⃣ Chargement des données existantes...")
    
    # Chargement de l'agent classique pour comparaison
    classical_agent = QLearningAgent(n_states, n_actions)
    try:
        classical_agent.load_agent(f"{results_dir}/q_learning_agent_classical.pkl")
        print("✅ Agent classique chargé")
    except FileNotFoundError:
        print("❌ Agent classique non trouvé. Entraînement d'abord...")
        return
    
    # Chargement des trajectoires de démonstration si disponibles
    demo_trajectories = []
    demo_preferences = []
    
    try:
        trajectory_manager.load_trajectories(f"{results_dir}/demo_trajectories.pkl")
        demo_trajectories = trajectory_manager.trajectories
        print(f"✅ {len(demo_trajectories)} trajectoires de démo chargées")
    except FileNotFoundError:
        print("⚠️ Pas de trajectoires de démo trouvées")
    
    try:
        with open(f"{results_dir}/demo_preferences.json", 'r') as f:
            pref_data = json.load(f)
            demo_preferences = pref_data['preferences']
        print(f"✅ {len(demo_preferences)} préférences de démo chargées")
    except FileNotFoundError:
        print("⚠️ Pas de préférences de démo trouvées")
    
    print("\n2️⃣ Création et configuration de l'agent PbRL...")
    
    # Création de l'agent PbRL
    pbrl_agent = PreferenceBasedQLearning(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        preference_weight=0.3  # Poids modéré pour les préférences
    )
    
    print("✅ Agent PbRL créé")
    
    # Choix du mode d'entraînement
    print("\n3️⃣ Sélection du mode d'entraînement...")
    print("Choisissez le mode d'entraînement:")
    print("1 - Entraînement avec préférences existantes seulement")
    print("2 - Entraînement interactif (avec collecte de nouvelles préférences)")
    print("3 - Entraînement standard (sans préférences, pour comparaison)")
    
    mode = input("Votre choix (1/2/3): ").strip()
    
    if mode == "1":
        # Mode 1: Utiliser seulement les préférences existantes
        print("\n🎯 MODE 1: Entraînement avec préférences existantes")
        
        if not demo_preferences or not demo_trajectories:
            print("❌ Pas assez de données de préférences. Génération de trajectoires...")
            # Générer quelques trajectoires avec l'agent classique
            print("Génération de 20 trajectoires avec l'agent classique...")
            for i in range(20):
                traj = trajectory_manager.collect_trajectory(env, classical_agent)
                demo_trajectories.append(traj)
            
            print("Collecte rapide de quelques préférences...")
            # Sélectionner quelques paires pour une démonstration rapide
            pairs = []
            sorted_trajs = sorted(demo_trajectories, key=lambda t: t.total_reward, reverse=True)
            
            # Paire best vs worst
            if len(sorted_trajs) >= 2:
                pairs.append((sorted_trajs[0], sorted_trajs[-1]))
            
            # Paire du milieu
            mid = len(sorted_trajs) // 2
            if mid > 0 and mid < len(sorted_trajs) - 1:
                pairs.append((sorted_trajs[mid], sorted_trajs[mid + 1]))
            
            # Collecte interactive minimale
            demo_preferences = []
            for traj1, traj2 in pairs:
                pref_data = {
                    'trajectory_a_id': traj1.episode_id,
                    'trajectory_b_id': traj2.episode_id,
                    'trajectory_a_reward': traj1.total_reward,
                    'trajectory_b_reward': traj2.total_reward,
                    'trajectory_a_length': traj1.episode_length,
                    'trajectory_b_length': traj2.episode_length,
                    'trajectory_a_efficiency': traj1.total_reward / traj1.episode_length,
                    'trajectory_b_efficiency': traj2.total_reward / traj2.episode_length,
                    'choice': 1 if traj1.total_reward > traj2.total_reward else 2,  # Auto: préfère meilleure récompense
                    'reasoning': "Préférence automatique basée sur la récompense totale"
                }
                demo_preferences.append(pref_data)
        
        # Entraînement avec préférences
        pbrl_rewards = pbrl_agent.train_with_preferences(
            env, demo_trajectories, demo_preferences, episodes=10000
        )
        
    elif mode == "2":
        # Mode 2: Entraînement interactif
        print("\n🎯 MODE 2: Entraînement interactif avec nouvelles préférences")
        
        pbrl_rewards, iteration_summaries = pbrl_agent.interactive_training_loop(
            env, preference_interface, trajectory_manager,
            episodes_per_iteration=2000,
            max_iterations=3,
            trajectories_per_comparison=8
        )
        
    else:
        # Mode 3: Entraînement standard (pour comparaison)
        print("\n🎯 MODE 3: Entraînement standard (sans préférences)")
        
        pbrl_rewards = pbrl_agent.train(env, episodes=10000)
    
    print("\n4️⃣ Évaluation et comparaison des agents...")
    
    # Évaluation des deux agents
    print("Évaluation de l'agent classique...")
    classical_avg, classical_eval = classical_agent.evaluate(env, episodes=100)
    
    print("Évaluation de l'agent PbRL...")
    pbrl_avg, pbrl_eval = pbrl_agent.evaluate(env, episodes=100)
    
    # Sauvegarde de l'agent PbRL
    pbrl_agent.save_pbrl_agent(f"{results_dir}/pbrl_agent.pkl")
    
    print("\n5️⃣ Génération des analyses et comparaisons...")
    
    # Comparaison des performances
    create_comparison_analysis(classical_agent, pbrl_agent, classical_eval, pbrl_eval, results_dir)
    
    # Analyse de l'apprentissage par préférences si applicable
    if mode in ["1", "2"]:
        preference_summary = pbrl_agent.get_preference_learning_summary()
        print(f"\n📊 RÉSUMÉ DE L'APPRENTISSAGE PAR PRÉFÉRENCES:")
        print(f"   Mises à jour par préférences: {preference_summary.get('total_preference_updates', 0)}")
        print(f"   Poids des préférences utilisé: {preference_summary.get('preference_weight_used', 0)}")
    
    # Résumé final
    print("\n" + "="*80)
    print("🎉 COMPARAISON AGENT CLASSIQUE vs AGENT PbRL")
    print("="*80)
    print(f"Agent Classique - Récompense moyenne: {classical_avg:.2f}")
    print(f"Agent PbRL      - Récompense moyenne: {pbrl_avg:.2f}")
    
    improvement = ((pbrl_avg - classical_avg) / abs(classical_avg)) * 100 if classical_avg != 0 else 0
    if improvement > 0:
        print(f"🚀 Amélioration PbRL: +{improvement:.1f}%")
    elif improvement < 0:
        print(f"📉 Dégradation PbRL: {improvement:.1f}%")
    else:
        print("📊 Performance équivalente")
    
    print(f"\n📁 Tous les résultats sauvegardés dans '{results_dir}/'")

def create_comparison_analysis(classical_agent, pbrl_agent, classical_eval, pbrl_eval, results_dir):
    """Crée une analyse comparative complète"""
    
    # Graphiques de comparaison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Courbes d'apprentissage
    if classical_agent.training_rewards and pbrl_agent.training_rewards:
        # Moyenne mobile pour lisser
        window = 100
        classical_smooth = [np.mean(classical_agent.training_rewards[max(0, i-window):i+1]) 
                          for i in range(len(classical_agent.training_rewards))]
        pbrl_smooth = [np.mean(pbrl_agent.training_rewards[max(0, i-window):i+1]) 
                      for i in range(len(pbrl_agent.training_rewards))]
        
        axes[0, 0].plot(classical_smooth, label='Agent Classique', alpha=0.8)
        axes[0, 0].plot(pbrl_smooth, label='Agent PbRL', alpha=0.8)
        axes[0, 0].set_title('Courbes d\'apprentissage (moyenne mobile)')
        axes[0, 0].set_xlabel('Épisode')
        axes[0, 0].set_ylabel('Récompense moyenne')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution des performances en évaluation
    axes[0, 1].hist(classical_eval, bins=15, alpha=0.7, label='Agent Classique', color='blue')
    axes[0, 1].hist(pbrl_eval, bins=15, alpha=0.7, label='Agent PbRL', color='orange')
    axes[0, 1].axvline(np.mean(classical_eval), color='blue', linestyle='--', alpha=0.8)
    axes[0, 1].axvline(np.mean(pbrl_eval), color='orange', linestyle='--', alpha=0.8)
    axes[0, 1].set_title('Distribution des performances (Évaluation)')
    axes[0, 1].set_xlabel('Récompense par épisode')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Comparaison des statistiques
    classical_stats = {
        'Moyenne': np.mean(classical_eval),
        'Médiane': np.median(classical_eval),
        'Écart-type': np.std(classical_eval),
        'Min': np.min(classical_eval),
        'Max': np.max(classical_eval)
    }
    
    pbrl_stats = {
        'Moyenne': np.mean(pbrl_eval),
        'Médiane': np.median(pbrl_eval),
        'Écart-type': np.std(pbrl_eval),
        'Min': np.min(pbrl_eval),
        'Max': np.max(pbrl_eval)
    }
    
    stats_comparison = []
    for metric in classical_stats.keys():
        stats_comparison.append([metric, classical_stats[metric], pbrl_stats[metric]])
    
    # Tableau de comparaison
    axes[1, 0].axis('tight')
    axes[1, 0].axis('off')
    table = axes[1, 0].table(cellText=[[f"{row[0]}", f"{row[1]:.2f}", f"{row[2]:.2f}"] for row in stats_comparison],
                           colLabels=['Métrique', 'Classique', 'PbRL'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    axes[1, 0].set_title('Comparaison statistique détaillée')
    
    # 4. Box plot comparatif
    data_to_plot = [classical_eval, pbrl_eval]
    box_plot = axes[1, 1].boxplot(data_to_plot, labels=['Classique', 'PbRL'], patch_artist=True)
    
    # Couleurs pour les box plots
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 1].set_title('Box Plot - Comparaison des performances')
    axes[1, 1].set_ylabel('Récompense par épisode')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/comparison_classical_vs_pbrl.png", dpi=300, bbox_inches='tight')
    print(f"📊 Graphique de comparaison sauvegardé: {results_dir}/comparison_classical_vs_pbrl.png")
    plt.show()
    
    # Sauvegarde des statistiques détaillées
    comparison_data = {
        'classical_agent': {
            'evaluation_scores': [float(x) for x in classical_eval],
            'statistics': {k: float(v) for k, v in classical_stats.items()},
            'training_episodes': len(classical_agent.training_rewards) if classical_agent.training_rewards else 0
        },
        'pbrl_agent': {
            'evaluation_scores': [float(x) for x in pbrl_eval],
            'statistics': {k: float(v) for k, v in pbrl_stats.items()},
            'training_episodes': len(pbrl_agent.training_rewards) if pbrl_agent.training_rewards else 0,
            'preference_updates': int(getattr(pbrl_agent, 'preference_updates', 0))
        },
        'comparison': {
            'improvement_percentage': float(((pbrl_stats['Moyenne'] - classical_stats['Moyenne']) / abs(classical_stats['Moyenne'])) * 100),
            'statistical_significance': 'TODO: t-test',  # Pourrait être ajouté
            'winner': 'PbRL' if pbrl_stats['Moyenne'] > classical_stats['Moyenne'] else 'Classique'
        }
    }
    
    with open(f"{results_dir}/detailed_comparison.json", 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"📋 Analyse détaillée sauvegardée: {results_dir}/detailed_comparison.json")

if __name__ == "__main__":
    main()