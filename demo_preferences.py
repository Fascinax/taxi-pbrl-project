import gymnasium as gym
import numpy as np
from src.q_learning_agent import QLearningAgent
from src.trajectory_manager import TrajectoryManager, Trajectory
from src.preference_interface import PreferenceInterface
import pickle
import os

def main():
    """
    Script pour démontrer le système de comparaison de trajectoires et de préférences
    """
    
    print("=== DÉMONSTRATION DU SYSTÈME DE PRÉFÉRENCES ===")
    
    # Chargement de l'agent pré-entraîné
    print("\n1️⃣ Chargement de l'agent Q-Learning entraîné...")
    agent = QLearningAgent(n_states=500, n_actions=6)
    
    try:
        agent.load_agent("results/q_learning_agent_classical.pkl")
        print("✅ Agent chargé avec succès!")
    except FileNotFoundError:
        print("❌ Agent non trouvé. Veuillez d'abord entraîner l'agent avec train_classical_agent.py")
        return
    
    # Création de l'environnement
    env = gym.make("Taxi-v3")
    
    # Initialisation des managers
    trajectory_manager = TrajectoryManager()
    preference_interface = PreferenceInterface()
    
    # Collecte de trajectoires
    print("\n2️⃣ Collecte de trajectoires de démonstration...")
    trajectories = []
    
    print("Génération de 10 trajectoires...")
    for i in range(10):
        traj = trajectory_manager.collect_trajectory(env, agent, max_steps=200, render=False)
        trajectories.append(traj)
        print(f"  Trajectoire {i+1}: Récompense = {traj.total_reward}, Longueur = {traj.episode_length}")
    
    # Sauvegarde des trajectoires
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    trajectory_manager.save_trajectories(f"{results_dir}/demo_trajectories.pkl")
    
    # Sélection de paires intéressantes pour comparaison
    print("\n3️⃣ Sélection de paires de trajectoires pour comparaison...")
    
    # Tri des trajectoires par récompense pour créer des paires intéressantes
    trajectories_sorted = sorted(trajectories, key=lambda t: t.total_reward, reverse=True)
    
    # Création de paires contrastées
    interesting_pairs = []
    
    # Paire 1: Meilleure vs moins bonne
    if len(trajectories_sorted) >= 2:
        interesting_pairs.append((trajectories_sorted[0], trajectories_sorted[-1]))
    
    # Paire 2: Deux trajectoires avec récompenses similaires mais longueurs différentes
    middle_trajectories = trajectories_sorted[2:6] if len(trajectories_sorted) > 5 else trajectories_sorted[1:3]
    if len(middle_trajectories) >= 2:
        # Trier par longueur
        middle_by_length = sorted(middle_trajectories, key=lambda t: t.episode_length)
        if len(middle_by_length) >= 2:
            interesting_pairs.append((middle_by_length[0], middle_by_length[-1]))
    
    # Paire 3: Deux trajectoires aléatoires du milieu
    if len(trajectories_sorted) >= 4:
        mid_idx = len(trajectories_sorted) // 2
        interesting_pairs.append((trajectories_sorted[mid_idx], trajectories_sorted[mid_idx + 1]))
    
    print(f"✅ {len(interesting_pairs)} paires sélectionnées pour comparaison")
    
    # Démonstration de la comparaison
    print("\n4️⃣ Démonstration de la comparaison de trajectoires...")
    
    if interesting_pairs:
        print("\n🔍 EXEMPLE DE COMPARAISON AUTOMATIQUE:")
        traj1, traj2 = interesting_pairs[0]
        trajectory_manager.display_trajectory_comparison(traj1, traj2)
        
        # Visualisation graphique
        print("\n📊 Génération de la visualisation graphique...")
        trajectory_manager.visualize_trajectories(
            traj1, traj2, 
            save_path=f"{results_dir}/trajectory_comparison_demo.png"
        )
    
    # Interface de préférences (mode démonstration)
    print("\n5️⃣ Démonstration de l'interface de préférences...")
    print("\n🤖 COLLECTE INTERACTIVE DE PRÉFÉRENCES")
    print("Vous allez maintenant pouvoir comparer des trajectoires et exprimer vos préférences.")
    
    demo_choice = input("\n👉 Voulez-vous tester l'interface de préférences ? (y/n): ").strip().lower()
    
    if demo_choice in ['y', 'yes', 'oui', 'o']:
        # Session de préférences interactive avec visualisation Gymnasium
        print("\n🎯 SESSION DE PRÉFÉRENCES INTERACTIVE")
        collected_preferences = preference_interface.collect_preference_batch(
            interesting_pairs[:2],  # Limiter à 2 comparaisons pour la démo
            trajectory_manager
        )
        
        # Sauvegarde et statistiques
        preference_interface.save_preferences(f"{results_dir}/demo_preferences.json")
        preference_interface.display_preferences_summary()
        
        print("\n✅ Démonstration des préférences terminée!")
        
    else:
        print("⏭️ Interface de préférences ignorée.")
    
    # Résumé et prochaines étapes
    print("\n" + "="*80)
    print("🎉 DÉMONSTRATION TERMINÉE")
    print("="*80)
    print("✅ Système de trajectoires opérationnel")
    print("✅ Interface de préférences fonctionnelle")
    print("✅ Visualisations et comparaisons disponibles")
    
    print(f"\n📁 Fichiers générés dans '{results_dir}/':")
    generated_files = [
        "demo_trajectories.pkl (trajectoires collectées)",
        "trajectory_comparison_demo.png (visualisation)",
    ]
    
    if demo_choice in ['y', 'yes', 'oui', 'o']:
        generated_files.append("demo_preferences.json (préférences collectées)")
    
    for file in generated_files:
        print(f"   • {file}")
    
    print(f"\n🚀 PROCHAINES ÉTAPES:")
    print("   1. Créer le système de conversion préférences → apprentissage")
    print("   2. Implémenter l'agent PbRL (Preference-based RL)")
    print("   3. Comparer agent classique vs agent PbRL")
    print("   4. Analyser les résultats et rédiger le rapport")
    
    env.close()

if __name__ == "__main__":
    main()