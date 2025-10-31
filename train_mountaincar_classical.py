"""
Script d'entraînement de l'agent Q-Learning classique sur MountainCar-v0
Comparable au train_classical_agent.py pour Taxi
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
from src.mountain_car_agent import MountainCarAgent


def plot_training_results(agent: MountainCarAgent, save_path: str = None):
    """
    Visualise les résultats d'entraînement
    
    Args:
        agent: Agent entraîné
        save_path: Chemin pour sauvegarder le graphique
    """
    rewards = agent.training_rewards
    
    # Calcul des moyennes mobiles
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    else:
        moving_avg = rewards
    
    # Création du graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Graphique 1: Récompenses au fil du temps
    ax1.plot(rewards, alpha=0.3, label='Récompense par épisode', color='lightblue')
    if len(rewards) >= window:
        ax1.plot(range(window-1, len(rewards)), moving_avg, 
                label=f'Moyenne mobile ({window} épisodes)', 
                color='darkblue', linewidth=2)
    ax1.axhline(y=-110, color='red', linestyle='--', 
               label='Objectif optimal (~-110)', alpha=0.7)
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense totale')
    ax1.set_title('Progression de l\'entraînement - MountainCar Q-Learning')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Distribution des récompenses finales
    final_rewards = rewards[-1000:] if len(rewards) >= 1000 else rewards
    ax2.hist(final_rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=np.mean(final_rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Moyenne: {np.mean(final_rewards):.2f}')
    ax2.set_xlabel('Récompense totale')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des récompenses (1000 derniers épisodes)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Graphique sauvegardé: {save_path}")
    
    plt.close()  # Ferme sans afficher pour éviter l'interruption


def plot_evaluation_results(rewards: list, stats: dict, save_path: str = None):
    """
    Visualise les résultats d'évaluation
    
    Args:
        rewards: Liste des récompenses
        stats: Statistiques d'évaluation
        save_path: Chemin pour sauvegarder le graphique
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramme des récompenses
    ax1.hist(rewards, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax1.axvline(x=stats['mean_reward'], color='red', linestyle='--', 
               linewidth=2, label=f"Moyenne: {stats['mean_reward']:.2f}")
    ax1.axvline(x=stats['mean_reward'] - stats['std_reward'], 
               color='orange', linestyle=':', alpha=0.7, 
               label=f"±1 std: [{stats['mean_reward']-stats['std_reward']:.2f}, "
                     f"{stats['mean_reward']+stats['std_reward']:.2f}]")
    ax1.axvline(x=stats['mean_reward'] + stats['std_reward'], 
               color='orange', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Récompense totale')
    ax1.set_ylabel('Fréquence')
    ax1.set_title(f"Distribution des récompenses\n({stats['total_episodes']} épisodes)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot([rewards], labels=['MountainCar Agent'])
    ax2.set_ylabel('Récompense totale')
    ax2.set_title('Statistiques des récompenses')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Ajout de texte avec statistiques
    stats_text = f"Moyenne: {stats['mean_reward']:.2f}\n"
    stats_text += f"Std: {stats['std_reward']:.2f}\n"
    stats_text += f"Min/Max: {stats['min_reward']:.0f}/{stats['max_reward']:.0f}\n"
    stats_text += f"Succès: {stats['success_rate']:.1f}%"
    ax2.text(1.15, stats['mean_reward'], stats_text, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Graphique sauvegardé: {save_path}")
    
    plt.close()  # Ferme sans afficher pour éviter l'interruption


def main():
    """Script principal d'entraînement"""
    
    print(f"\n{'='*80}")
    print("🚗 ENTRAÎNEMENT AGENT Q-LEARNING CLASSIQUE - MOUNTAINCAR-V0")
    print(f"{'='*80}\n")
    
    # Configuration
    TRAIN_EPISODES = 6000  # RÉDUIT pour avoir 70-80% succès
    EVAL_EPISODES = 200
    N_POSITION_BINS = 20
    N_VELOCITY_BINS = 20
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.99
    EPSILON_DECAY = 0.999
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    print("[CONFIG]  CONFIGURATION:")
    print(f"   - Épisodes d'entraînement: {TRAIN_EPISODES}")
    print(f"   - Épisodes d'évaluation: {EVAL_EPISODES}")
    print(f"   - Bins position: {N_POSITION_BINS}")
    print(f"   - Bins vitesse: {N_VELOCITY_BINS}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Gamma: {DISCOUNT_FACTOR}")
    print(f"   - Epsilon decay: {EPSILON_DECAY}")
    print()
    
    # Création de l'environnement
    print("[ENV] Création de l'environnement MountainCar-v0...")
    env = gym.make('MountainCar-v0')
    print("[OK] Environnement créé")
    print(f"   - Espace d'états: Position [-1.2, 0.6], Vitesse [-0.07, 0.07]")
    print(f"   - Actions: 3 (0=gauche, 1=rien, 2=droite)")
    print(f"   - Objectif: Atteindre position >= 0.5")
    print()
    
    # Création de l'agent
    print("[AGENT] Création de l'agent Q-Learning avec discrétisation...")
    agent = MountainCarAgent(
        n_position_bins=N_POSITION_BINS,
        n_velocity_bins=N_VELOCITY_BINS,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=1.0,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=0.01
    )
    print()
    
    # Entraînement
    print(f"[START] PHASE 1: ENTRAÎNEMENT ({TRAIN_EPISODES} épisodes)")
    print("-" * 80)
    start_time = datetime.now()
    
    training_rewards = agent.train(
        env=env,
        episodes=TRAIN_EPISODES,
        max_steps=200,
        verbose=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"[TIME]  Temps d'entraînement: {training_time:.2f} secondes")
    print()
    
    # Évaluation
    print(f"[PLOT] PHASE 2: ÉVALUATION ({EVAL_EPISODES} épisodes)")
    print("-" * 80)
    
    eval_rewards, eval_stats = agent.evaluate(
        env=env,
        episodes=EVAL_EPISODES,
        verbose=True
    )
    
    # Sauvegarde de l'agent
    agent_path = os.path.join(results_dir, "mountain_car_agent_classical.pkl")
    print(f"[SAVE] Sauvegarde de l'agent...")
    agent.save_agent(agent_path)
    
    # Visualisation de la politique apprise
    print("\n[MAP]  POLITIQUE APPRISE")
    print("-" * 80)
    agent.visualize_policy()
    
    # Création des graphiques
    print("[CHART] CRÉATION DES VISUALISATIONS")
    print("-" * 80)
    
    # Graphique d'entraînement
    training_plot_path = os.path.join(results_dir, "training_progress_mountaincar.png")
    plot_training_results(agent, training_plot_path)
    
    # Graphique d'évaluation
    eval_plot_path = os.path.join(results_dir, "evaluation_histogram_mountaincar.png")
    plot_evaluation_results(eval_rewards, eval_stats, eval_plot_path)
    
    # Sauvegarde des résultats en JSON
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'environment': 'MountainCar-v0',
        'training': {
            'episodes': TRAIN_EPISODES,
            'time_seconds': training_time,
            'final_avg_reward_100': float(np.mean(training_rewards[-100:])),
            'hyperparameters': {
                'n_position_bins': N_POSITION_BINS,
                'n_velocity_bins': N_VELOCITY_BINS,
                'learning_rate': LEARNING_RATE,
                'discount_factor': DISCOUNT_FACTOR,
                'epsilon_decay': EPSILON_DECAY
            }
        },
        'evaluation': {
            'episodes': EVAL_EPISODES,
            'mean_reward': float(eval_stats['mean_reward']),
            'std_reward': float(eval_stats['std_reward']),
            'min_reward': float(eval_stats['min_reward']),
            'max_reward': float(eval_stats['max_reward']),
            'mean_length': float(eval_stats['mean_length']),
            'std_length': float(eval_stats['std_length']),
            'success_rate': float(eval_stats['success_rate'])
        }
    }
    
    import json
    results_json_path = os.path.join(results_dir, "mountaincar_classical_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"[SAVE] Résultats sauvegardés: {results_json_path}")
    
    # Résumé final
    print(f"\n{'='*80}")
    print("[DONE] ENTRAÎNEMENT TERMINÉ - RÉSUMÉ")
    print(f"{'='*80}")
    print(f"Agent: Q-Learning Classique (MountainCar-v0)")
    print(f"Entraînement: {TRAIN_EPISODES} épisodes en {training_time:.2f}s")
    print(f"Récompense moyenne (100 derniers): {np.mean(training_rewards[-100:]):.2f}")
    print(f"\nÉvaluation ({EVAL_EPISODES} épisodes):")
    print(f"  - Récompense: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"  - Longueur: {eval_stats['mean_length']:.1f} ± {eval_stats['std_length']:.1f} pas")
    print(f"  - Taux de succès: {eval_stats['success_rate']:.1f}%")
    print(f"\nFichiers générés:")
    print(f"  - Agent: {agent_path}")
    print(f"  - Résultats: {results_json_path}")
    print(f"  - Graphique entraînement: {training_plot_path}")
    print(f"  - Graphique évaluation: {eval_plot_path}")
    print(f"{'='*80}\n")
    
    print("[OK] Prochaines étapes:")
    print("   1. Exécuter: python demo_mountaincar.py (voir l'agent en action)")
    print("   2. Exécuter: python collect_mountaincar_preferences.py (collecter préférences)")
    print("   3. Exécuter: python train_mountaincar_pbrl.py (entraîner agent PbRL)")
    print()
    
    env.close()


if __name__ == "__main__":
    main()
