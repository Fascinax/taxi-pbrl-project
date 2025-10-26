"""
Script de démonstration de MountainCar-v0
Teste l'environnement et visualise un agent entraîné
"""

import gymnasium as gym
import numpy as np
import os
import time
from src.mountain_car_agent import MountainCarAgent


def watch_random_agent(episodes: int = 3):
    """
    Observe un agent aléatoire sur MountainCar
    
    Args:
        episodes: Nombre d'épisodes à observer
    """
    print(f"\n{'='*80}")
    print("🎲 AGENT ALÉATOIRE - MOUNTAINCAR-V0")
    print(f"{'='*80}\n")
    
    env = gym.make('MountainCar-v0', render_mode='human')
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n🎬 Épisode {episode + 1}/{episodes}")
        print(f"Position initiale: {state[0]:.3f}, Vitesse initiale: {state[1]:.3f}")
        
        while steps < 200:
            action = env.action_space.sample()  # Action aléatoire
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        success = state[0] >= 0.5
        print(f"Résultat: {'✅ Succès!' if success else '❌ Échec'}")
        print(f"Position finale: {state[0]:.3f}")
        print(f"Récompense totale: {total_reward:.0f}")
        print(f"Nombre de pas: {steps}")
    
    env.close()
    print(f"\n{'='*80}\n")


def watch_trained_agent(agent_path: str, episodes: int = 5):
    """
    Observe un agent entraîné sur MountainCar
    
    Args:
        agent_path: Chemin vers l'agent sauvegardé
        episodes: Nombre d'épisodes à observer
    """
    print(f"\n{'='*80}")
    print("🤖 AGENT ENTRAÎNÉ - MOUNTAINCAR-V0")
    print(f"{'='*80}\n")
    
    if not os.path.exists(agent_path):
        print(f"❌ Agent non trouvé: {agent_path}")
        print("💡 Exécutez d'abord: python train_mountaincar_classical.py")
        return
    
    # Chargement de l'agent
    print(f"📂 Chargement de l'agent: {agent_path}")
    agent = MountainCarAgent()
    agent.load_agent(agent_path)
    print("✅ Agent chargé\n")
    
    env = gym.make('MountainCar-v0', render_mode='human')
    
    successes = 0
    total_rewards = []
    total_steps_list = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n🎬 Épisode {episode + 1}/{episodes}")
        print(f"Position initiale: {state[0]:.3f}, Vitesse initiale: {state[1]:.3f}")
        
        # Analyse de la stratégie initiale
        analysis = agent.get_state_analysis(state)
        print(f"Première action recommandée: {analysis['best_action_name']}")
        
        while steps < 200:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Ralentir un peu pour mieux voir
            time.sleep(0.01)
            
            if done:
                break
        
        success = state[0] >= 0.5
        if success:
            successes += 1
        
        total_rewards.append(total_reward)
        total_steps_list.append(steps)
        
        print(f"Résultat: {'✅ Succès!' if success else '❌ Échec'}")
        print(f"Position finale: {state[0]:.3f}")
        print(f"Récompense totale: {total_reward:.0f}")
        print(f"Nombre de pas: {steps}")
    
    env.close()
    
    # Statistiques finales
    print(f"\n{'='*80}")
    print("📊 STATISTIQUES")
    print(f"{'='*80}")
    print(f"Taux de succès: {(successes/episodes)*100:.1f}% ({successes}/{episodes})")
    print(f"Récompense moyenne: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Pas moyens: {np.mean(total_steps_list):.1f} ± {np.std(total_steps_list):.1f}")
    print(f"{'='*80}\n")


def analyze_environment():
    """Analyse détaillée de l'environnement MountainCar"""
    print(f"\n{'='*80}")
    print("🔬 ANALYSE DE L'ENVIRONNEMENT MOUNTAINCAR-V0")
    print(f"{'='*80}\n")
    
    env = gym.make('MountainCar-v0')
    
    print("📋 INFORMATIONS GÉNÉRALES")
    print("-" * 80)
    print(f"Espace d'observation: {env.observation_space}")
    print(f"  - Position: [-1.2, 0.6]")
    print(f"  - Vitesse: [-0.07, 0.07]")
    print(f"\nEspace d'actions: {env.action_space}")
    print(f"  - 0: Accélère vers la gauche")
    print(f"  - 1: Pas d'accélération")
    print(f"  - 2: Accélère vers la droite")
    print(f"\nObjectif: Atteindre position >= 0.5 (drapeau)")
    print(f"Récompense: -1 à chaque pas (encourage rapidité)")
    print(f"Max steps: 200")
    
    print(f"\n{'='*80}")
    print("🧪 TEST DE L'ENVIRONNEMENT")
    print(f"{'='*80}\n")
    
    # Test de quelques actions
    state, _ = env.reset()
    print(f"État initial: position={state[0]:.3f}, vitesse={state[1]:.3f}")
    
    actions = [2, 2, 2, 2, 2]  # Accélère vers la droite
    print(f"\nTest: 5 actions vers la droite")
    for i, action in enumerate(actions):
        state, reward, terminated, truncated, _ = env.step(action)
        print(f"  Pas {i+1}: position={state[0]:.3f}, vitesse={state[1]:.3f}, "
              f"reward={reward:.0f}")
        if terminated or truncated:
            break
    
    # Reset et test actions gauche
    state, _ = env.reset()
    actions = [0, 0, 0, 0, 0]  # Accélère vers la gauche
    print(f"\nTest: 5 actions vers la gauche")
    for i, action in enumerate(actions):
        state, reward, terminated, truncated, _ = env.step(action)
        print(f"  Pas {i+1}: position={state[0]:.3f}, vitesse={state[1]:.3f}, "
              f"reward={reward:.0f}")
        if terminated or truncated:
            break
    
    print(f"\n💡 STRATÉGIE OPTIMALE:")
    print("-" * 80)
    print("La voiture doit prendre de l'élan en oscillant entre les deux collines")
    print("pour accumuler assez de vitesse et atteindre le drapeau.")
    print("Actions directes vers la droite ne suffisent généralement pas!")
    print(f"{'='*80}\n")
    
    env.close()


def compare_strategies():
    """Compare différentes stratégies simples"""
    print(f"\n{'='*80}")
    print("⚖️  COMPARAISON DE STRATÉGIES SIMPLES")
    print(f"{'='*80}\n")
    
    env = gym.make('MountainCar-v0')
    
    strategies = {
        'Toujours droite': lambda state: 2,
        'Toujours gauche': lambda state: 0,
        'Selon vitesse': lambda state: 2 if state[1] >= 0 else 0,
        'Oscillation': lambda state: 2 if state[0] >= -0.5 else 0,
    }
    
    for name, strategy in strategies.items():
        rewards = []
        successes = 0
        
        for _ in range(10):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 200:
                action = strategy(state)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            rewards.append(total_reward)
            if state[0] >= 0.5:
                successes += 1
        
        print(f"📊 {name:20} | Succès: {successes:2}/10 | "
              f"Récompense moy: {np.mean(rewards):7.2f}")
    
    print(f"\n{'='*80}\n")
    env.close()


def main():
    """Script principal de démonstration"""
    print(f"\n{'='*80}")
    print("🎮 DÉMONSTRATION MOUNTAINCAR-V0")
    print(f"{'='*80}\n")
    
    print("Choisissez une option:")
    print("1. Analyser l'environnement")
    print("2. Voir un agent aléatoire")
    print("3. Voir l'agent entraîné")
    print("4. Comparer des stratégies simples")
    print("5. Tout exécuter")
    print("0. Quitter")
    
    choice = input("\nVotre choix (0-5): ").strip()
    
    if choice == '1':
        analyze_environment()
    elif choice == '2':
        watch_random_agent(episodes=3)
    elif choice == '3':
        agent_path = "results/mountain_car_agent_classical.pkl"
        watch_trained_agent(agent_path, episodes=5)
    elif choice == '4':
        compare_strategies()
    elif choice == '5':
        analyze_environment()
        input("\nAppuyez sur Entrée pour voir l'agent aléatoire...")
        watch_random_agent(episodes=2)
        input("\nAppuyez sur Entrée pour comparer les stratégies...")
        compare_strategies()
        agent_path = "results/mountain_car_agent_classical.pkl"
        if os.path.exists(agent_path):
            input("\nAppuyez sur Entrée pour voir l'agent entraîné...")
            watch_trained_agent(agent_path, episodes=3)
        else:
            print("\n⚠️  Agent entraîné non trouvé. Exécutez d'abord:")
            print("   python train_mountaincar_classical.py")
    elif choice == '0':
        print("Au revoir! 👋")
    else:
        print("❌ Choix invalide")


if __name__ == "__main__":
    main()
