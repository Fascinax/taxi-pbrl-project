"""
Script de test rapide pour la visualisation graphique des préférences
"""

import gymnasium as gym
from src.q_learning_agent import QLearningAgent
from src.trajectory_manager import TrajectoryManager
from src.preference_interface import PreferenceInterface
import pickle
import os

def test_visual_preference():
    """Test de la visualisation graphique des préférences"""
    
    print("🔧 Initialisation de l'environnement Taxi-v3...")
    env = gym.make("Taxi-v3")
    
    # Charger un agent entraîné ou en créer un simple
    agent_path = "results/q_learning_agent_classical.pkl"
    
    if os.path.exists(agent_path):
        print(f"📦 Chargement de l'agent depuis {agent_path}")
        with open(agent_path, 'rb') as f:
            agent_data = pickle.load(f)
        
        agent = QLearningAgent(env.observation_space.n, env.action_space.n)
        agent.q_table = agent_data['q_table']
        agent.epsilon = 0.0  # Mode greedy pour la collecte
    else:
        print("⚠️ Aucun agent sauvegardé trouvé. Création d'un agent simple...")
        agent = QLearningAgent(env.observation_space.n, env.action_space.n)
        agent.epsilon = 0.1
        # Entraînement rapide
        print("🏋️ Entraînement rapide (500 épisodes)...")
        agent.train(env, episodes=500)
        agent.epsilon = 0.0
    
    # Collecte de trajectoires
    print("\n🎬 Collecte de deux trajectoires pour comparaison...")
    traj_manager = TrajectoryManager()
    
    traj1 = traj_manager.collect_trajectory(env, agent, max_steps=200)
    traj2 = traj_manager.collect_trajectory(env, agent, max_steps=200)
    
    print(f"✅ Trajectoire A: {traj1.total_reward:.1f} points en {traj1.episode_length} pas")
    print(f"✅ Trajectoire B: {traj2.total_reward:.1f} points en {traj2.episode_length} pas")
    
    # Test de l'interface de préférences avec visualisation
    print("\n🎯 Lancement de l'interface de préférences avec visualisation Gymnasium...")
    preference_interface = PreferenceInterface()
    
    try:
        choice = preference_interface.collect_preference_interactive(
            traj1, traj2, traj_manager, use_visual=True
        )
        
        print(f"\n✨ Vous avez choisi: {['Égalité', 'Trajectoire A', 'Trajectoire B'][choice]}")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
    
    env.close()
    print("\n✅ Test terminé!")

if __name__ == "__main__":
    test_visual_preference()
