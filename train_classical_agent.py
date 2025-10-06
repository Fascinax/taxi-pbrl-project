import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from src.q_learning_agent import QLearningAgent
import os

def main():
    """
    Script principal pour entraîner et évaluer l'agent Q-Learning sur Taxi-v3
    """
    
    print("=== Entraînement Agent Q-Learning sur Taxi-v3 ===")
    
    # Création de l'environnement
    env = gym.make("Taxi-v3")
    
    # Paramètres de l'environnement
    n_states = env.observation_space.n  # 500 états
    n_actions = env.action_space.n      # 6 actions
    
    print(f"Environnement: {n_states} états, {n_actions} actions")
    
    # Création de l'agent
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Entraînement
    print("\n--- Début de l'entraînement ---")
    episodes = 15000
    training_rewards = agent.train(env, episodes=episodes)
    
    # Sauvegarde de l'agent
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    agent.save_agent(f"{results_dir}/q_learning_agent_classical.pkl")
    
    # Évaluation
    print("\n--- Évaluation de l'agent entraîné ---")
    avg_reward, eval_rewards = agent.evaluate(env, episodes=100)
    
    # Création de graphiques
    print("\n--- Génération des graphiques ---")
    agent.plot_training_progress(save_path=f"{results_dir}/training_progress_classical.png")
    
    # Graphique des résultats d'évaluation
    plt.figure(figsize=(10, 6))
    plt.hist(eval_rewards, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(avg_reward, color='red', linestyle='--', 
                label=f'Moyenne: {avg_reward:.2f}')
    plt.xlabel('Récompense totale par épisode')
    plt.ylabel('Fréquence')
    plt.title('Distribution des récompenses (Évaluation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/evaluation_histogram_classical.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Démonstration avec rendu visuel
    print("\n--- Démonstration (5 épisodes avec rendu) ---")
    env_render = gym.make("Taxi-v3", render_mode="human")
    demo_rewards = []
    
    for episode in range(5):
        state, _ = env_render.reset()
        total_reward = 0
        steps = 0
        max_steps = 200
        
        print(f"\nÉpisode de démonstration {episode + 1}:")
        
        while steps < max_steps:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env_render.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        demo_rewards.append(total_reward)
        print(f"Récompense: {total_reward}, Pas: {steps}")
    
    env_render.close()
    
    # Résumé final
    print("\n=== RÉSUMÉ FINAL ===")
    print(f"Entraînement: {episodes} épisodes")
    print(f"Récompense finale moyenne (100 derniers épisodes): {np.mean(training_rewards[-100:]):.2f}")
    print(f"Récompense d'évaluation moyenne: {avg_reward:.2f}")
    print(f"Récompense de démonstration moyenne: {np.mean(demo_rewards):.2f}")
    print(f"Fichiers sauvegardés dans le dossier '{results_dir}/'")

if __name__ == "__main__":
    main()