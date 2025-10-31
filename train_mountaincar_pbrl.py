"""
Script d'entraînement PBRL pour MountainCar-v0
Entraîne un agent avec préférences et compare avec l'agent classique
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
from datetime import datetime
from src.mountain_car_pbrl_agent import MountainCarPbRLAgent
from src.mountain_car_agent import MountainCarAgent
from collect_mountaincar_preferences import MountainCarTrajectory


def plot_comparison(classical_rewards, pbrl_rewards, save_path=None):
    """
    Compare les courbes d'apprentissage
    
    Args:
        classical_rewards: Récompenses agent classique
        pbrl_rewards: Récompenses agent PBRL
        save_path: Chemin de sauvegarde
    """
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Moyennes mobiles
    window = 100
    if len(classical_rewards) >= window:
        classical_ma = np.convolve(classical_rewards, np.ones(window)/window, mode='valid')
    else:
        classical_ma = classical_rewards
    
    if len(pbrl_rewards) >= window:
        pbrl_ma = np.convolve(pbrl_rewards, np.ones(window)/window, mode='valid')
    else:
        pbrl_ma = pbrl_rewards
    
    # Graphique 1: Courbes d'apprentissage
    ax1.plot(classical_rewards, alpha=0.2, color='blue', label='Classique (brut)')
    ax1.plot(pbrl_rewards, alpha=0.2, color='red', label='PBRL (brut)')
    
    if len(classical_ma) > 0:
        ax1.plot(range(window-1, len(classical_rewards)), classical_ma, 
                color='darkblue', linewidth=2, label='Classique (MA 100)')
    if len(pbrl_ma) > 0:
        ax1.plot(range(window-1, len(pbrl_rewards)), pbrl_ma,
                color='darkred', linewidth=2, label='PBRL (MA 100)')
    
    ax1.axhline(y=-110, color='green', linestyle='--', 
               label='Objectif optimal (~-110)', alpha=0.7)
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense totale')
    ax1.set_title('Comparaison MountainCar: Classique vs PBRL')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Distribution finale
    classical_final = classical_rewards[-1000:] if len(classical_rewards) >= 1000 else classical_rewards
    pbrl_final = pbrl_rewards[-1000:] if len(pbrl_rewards) >= 1000 else pbrl_rewards
    
    ax2.hist(classical_final, bins=30, alpha=0.5, color='blue', 
            label=f'Classique (μ={np.mean(classical_final):.2f})', edgecolor='black')
    ax2.hist(pbrl_final, bins=30, alpha=0.5, color='red',
            label=f'PBRL (μ={np.mean(pbrl_final):.2f})', edgecolor='black')
    ax2.set_xlabel('Récompense totale')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des récompenses (1000 derniers épisodes)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Graphique sauvegardé: {save_path}")
    
    plt.close()


def main():
    """Script principal d'entraînement PBRL"""
    
    print(f"\n{'='*80}")
    print("[START] ENTRAÎNEMENT PBRL - MOUNTAINCAR-V0")
    print(f"{'='*80}\n")
    
    # Configuration pour comparaison équitable
    PBRL_EPISODES = 6000  # Même nombre que classical
    EVAL_EPISODES = 200
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Vérification des fichiers nécessaires
    preferences_path = os.path.join(results_dir, "mountaincar_preferences.json")
    trajectories_path = os.path.join(results_dir, "mountaincar_trajectories.pkl")
    classical_agent_path = os.path.join(results_dir, "mountain_car_agent_classical.pkl")
    
    if not os.path.exists(preferences_path):
        print(f"[ERROR] Préférences non trouvées: {preferences_path}")
        print("[INFO] Exécutez d'abord: python collect_mountaincar_preferences.py")
        return
    
    if not os.path.exists(trajectories_path):
        print(f"[ERROR] Trajectoires non trouvées: {trajectories_path}")
        print("[INFO] Exécutez d'abord: python collect_mountaincar_preferences.py")
        return
    
    if not os.path.exists(classical_agent_path):
        print(f"[ERROR] Agent classique non trouvé: {classical_agent_path}")
        print("[INFO] Exécutez d'abord: python train_mountaincar_classical.py")
        return
    
    # Chargement des données
    print("[LOAD] CHARGEMENT DES DONNÉES")
    print("-" * 80)
    
    with open(preferences_path, 'r') as f:
        preferences = json.load(f)
    print(f"[OK] {len(preferences)} préférences chargées")
    
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"[OK] {len(trajectories)} trajectoires chargées")
    
    # Chargement agent classique pour comparaison
    classical_agent = MountainCarAgent()
    classical_agent.load_agent(classical_agent_path)
    classical_rewards = classical_agent.training_rewards
    print(f"[OK] Agent classique chargé ({len(classical_rewards)} épisodes d'entraînement)\n")
    
    # Création de l'environnement
    print("[ENV] Création de l'environnement...")
    env = gym.make('MountainCar-v0')
    print("[OK] Environnement créé\n")
    
    # Création de l'agent PBRL avec paramètres ÉQUILIBRÉS
    print("[AGENT] CRÉATION DE L'AGENT PBRL")
    print("-" * 80)
    agent_pbrl = MountainCarPbRLAgent(
        n_position_bins=20,
        n_velocity_bins=20,
        learning_rate=0.12,        # Équilibré
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,       # Même que classical pour comparaison
        epsilon_min=0.01,
        preference_weight=0.6      # Modéré pour éviter sur-apprentissage
    )
    print()
    
    # Entraînement PBRL
    print(f"[TRAIN] PHASE 1: ENTRAÎNEMENT PBRL ({PBRL_EPISODES} épisodes)")
    print("-" * 80)
    start_time = datetime.now()
    
    pbrl_rewards = agent_pbrl.train_with_preferences(
        env=env,
        trajectories=trajectories,
        preferences=preferences,
        episodes=PBRL_EPISODES
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"[TIME] Temps d'entraînement: {training_time:.2f} secondes\n")
    
    # Évaluation PBRL
    print(f"[EVAL] PHASE 2: ÉVALUATION PBRL ({EVAL_EPISODES} épisodes)")
    print("-" * 80)
    
    pbrl_eval_rewards, pbrl_eval_stats = agent_pbrl.evaluate(
        env=env,
        episodes=EVAL_EPISODES,
        verbose=True
    )
    
    # Évaluation agent classique
    print(f"[PLOT] PHASE 3: ÉVALUATION AGENT CLASSIQUE ({EVAL_EPISODES} épisodes)")
    print("-" * 80)
    
    classical_eval_rewards, classical_eval_stats = classical_agent.evaluate(
        env=env,
        episodes=EVAL_EPISODES,
        verbose=True
    )
    
    # Sauvegarde de l'agent PBRL
    pbrl_agent_path = os.path.join(results_dir, "mountain_car_agent_pbrl.pkl")
    agent_pbrl.save_pbrl_agent(pbrl_agent_path)
    
    # Visualisation de la politique PBRL
    print("\n[MAP]  POLITIQUE APPRISE (PBRL)")
    print("-" * 80)
    agent_pbrl.visualize_policy()
    
    # Comparaison graphique
    print("[CHART] CRÉATION DES VISUALISATIONS")
    print("-" * 80)
    
    comparison_plot_path = os.path.join(results_dir, "comparison_mountaincar_classical_vs_pbrl.png")
    plot_comparison(classical_rewards, pbrl_rewards, comparison_plot_path)
    
    # Analyse comparative détaillée
    print(f"\n{'='*80}")
    print("[PLOT] ANALYSE COMPARATIVE DÉTAILLÉE")
    print(f"{'='*80}\n")
    
    # Comparaison training
    print("[TRAIN]  ENTRAÎNEMENT")
    print("-" * 80)
    print(f"Classique:")
    print(f"  - Épisodes: {len(classical_rewards)}")
    print(f"  - Récompense finale (100 derniers): {np.mean(classical_rewards[-100:]):.2f}")
    print(f"\nPBRL:")
    print(f"  - Épisodes: {len(pbrl_rewards)}")
    print(f"  - Préférences utilisées: {len(preferences)}")
    print(f"  - Récompense finale (100 derniers): {np.mean(pbrl_rewards[-100:]):.2f}")
    
    efficiency = ((len(classical_rewards) - len(pbrl_rewards)) / len(classical_rewards)) * 100
    print(f"\n  → Efficacité PBRL: {efficiency:.1f}% moins d'épisodes")
    
    # Comparaison évaluation
    print(f"\n[PLOT] ÉVALUATION ({EVAL_EPISODES} épisodes)")
    print("-" * 80)
    print(f"{'Métrique':<25} {'Classique':<15} {'PBRL':<15} {'Différence'}")
    print("-" * 80)
    
    metrics = [
        ('Récompense moyenne', classical_eval_stats['mean_reward'], pbrl_eval_stats['mean_reward']),
        ('Écart-type', classical_eval_stats['std_reward'], pbrl_eval_stats['std_reward']),
        ('Récompense min', classical_eval_stats['min_reward'], pbrl_eval_stats['min_reward']),
        ('Récompense max', classical_eval_stats['max_reward'], pbrl_eval_stats['max_reward']),
        ('Longueur moyenne', classical_eval_stats['mean_length'], pbrl_eval_stats['mean_length']),
        ('Taux de succès (%)', classical_eval_stats['success_rate'], pbrl_eval_stats['success_rate'])
    ]
    
    for metric_name, classical_val, pbrl_val in metrics:
        diff = pbrl_val - classical_val
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        print(f"{metric_name:<25} {classical_val:<15.2f} {pbrl_val:<15.2f} {diff_str}")
    
    # Sauvegarde des résultats
    comparison_results = {
        'timestamp': datetime.now().isoformat(),
        'environment': 'MountainCar-v0',
        'training': {
            'classical_episodes': len(classical_rewards),
            'pbrl_episodes': len(pbrl_rewards),
            'pbrl_preferences_used': len(preferences),
            'efficiency_gain_percent': efficiency,
            'classical_final_reward_100': float(np.mean(classical_rewards[-100:])),
            'pbrl_final_reward_100': float(np.mean(pbrl_rewards[-100:])),
            'pbrl_training_time_seconds': training_time
        },
        'evaluation': {
            'episodes': EVAL_EPISODES,
            'classical': {
                'mean_reward': float(classical_eval_stats['mean_reward']),
                'std_reward': float(classical_eval_stats['std_reward']),
                'min_reward': float(classical_eval_stats['min_reward']),
                'max_reward': float(classical_eval_stats['max_reward']),
                'mean_length': float(classical_eval_stats['mean_length']),
                'success_rate': float(classical_eval_stats['success_rate'])
            },
            'pbrl': {
                'mean_reward': float(pbrl_eval_stats['mean_reward']),
                'std_reward': float(pbrl_eval_stats['std_reward']),
                'min_reward': float(pbrl_eval_stats['min_reward']),
                'max_reward': float(pbrl_eval_stats['max_reward']),
                'mean_length': float(pbrl_eval_stats['mean_length']),
                'success_rate': float(pbrl_eval_stats['success_rate'])
            },
            'improvements': {
                'reward_diff': float(pbrl_eval_stats['mean_reward'] - classical_eval_stats['mean_reward']),
                'length_diff': float(pbrl_eval_stats['mean_length'] - classical_eval_stats['mean_length']),
                'success_rate_diff': float(pbrl_eval_stats['success_rate'] - classical_eval_stats['success_rate'])
            }
        }
    }
    
    results_json_path = os.path.join(results_dir, "mountaincar_pbrl_comparison.json")
    with open(results_json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\n[SAVE] Résultats sauvegardés: {results_json_path}")
    
    # Résumé final
    print(f"\n{'='*80}")
    print("[DONE] ENTRAÎNEMENT PBRL TERMINÉ - RÉSUMÉ")
    print(f"{'='*80}")
    
    reward_improvement = pbrl_eval_stats['mean_reward'] - classical_eval_stats['mean_reward']
    reward_improvement_pct = (reward_improvement / abs(classical_eval_stats['mean_reward'])) * 100
    
    print(f"\n[CHART] PERFORMANCES")
    print("-" * 80)
    print(f"Classique: {classical_eval_stats['mean_reward']:.2f} ± {classical_eval_stats['std_reward']:.2f}")
    print(f"PBRL:      {pbrl_eval_stats['mean_reward']:.2f} ± {pbrl_eval_stats['std_reward']:.2f}")
    
    if reward_improvement > 0:
        print(f"\n[OK] PBRL est MEILLEUR: +{reward_improvement:.2f} ({reward_improvement_pct:+.2f}%)")
    else:
        print(f"\n[WARN]  Classique est meilleur: {reward_improvement:.2f} ({reward_improvement_pct:.2f}%)")
    
    print(f"\n[FAST] EFFICACITÉ D'APPRENTISSAGE")
    print("-" * 80)
    print(f"PBRL utilise {efficiency:.1f}% moins d'épisodes pour atteindre des performances similaires")
    
    print(f"\n[FILES] FICHIERS GÉNÉRÉS")
    print("-" * 80)
    print(f"  - Agent PBRL: {pbrl_agent_path}")
    print(f"  - Résultats: {results_json_path}")
    print(f"  - Graphique: {comparison_plot_path}")
    print(f"{'='*80}\n")
    
    env.close()


if __name__ == "__main__":
    main()
