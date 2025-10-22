"""
Collection automatique de préférences pour MountainCar (mode simulation)
Pour tester rapidement le workflow PBRL sans interaction manuelle
"""

import gymnasium as gym
import numpy as np
import pickle
import json
import os
from datetime import datetime
from src.mountain_car_agent import MountainCarAgent
from collect_mountaincar_preferences import collect_mountaincar_trajectory, MountainCarTrajectory


def auto_select_preference(traj_a: MountainCarTrajectory, traj_b: MountainCarTrajectory) -> int:
    """
    Sélection automatique basée sur des critères objectifs
    
    Returns:
        1 si A est meilleure, 2 si B est meilleure, 0 si égalité
    """
    # Critère 1: Succès vs échec
    if traj_a.success and not traj_b.success:
        return 1
    if traj_b.success and not traj_a.success:
        return 2
    
    # Critère 2: Si les deux réussissent, préférer la plus rapide
    if traj_a.success and traj_b.success:
        if abs(traj_a.episode_length - traj_b.episode_length) > 2:
            return 1 if traj_a.episode_length < traj_b.episode_length else 2
    
    # Critère 3: Si les deux échouent, préférer celle qui monte plus haut
    if not traj_a.success and not traj_b.success:
        position_diff = abs(traj_a.max_position - traj_b.max_position)
        if position_diff > 0.05:
            return 1 if traj_a.max_position > traj_b.max_position else 2
    
    # Critère 4: Différence de récompense significative
    reward_diff = abs(traj_a.total_reward - traj_b.total_reward)
    if reward_diff > 5:
        return 1 if traj_a.total_reward > traj_b.total_reward else 2
    
    # Sinon, égalité
    return 0


def main():
    """Collection automatique de préférences"""
    
    print(f"\n{'='*80}")
    print("🤖 COLLECTE AUTOMATIQUE DE PRÉFÉRENCES - MOUNTAINCAR-V0")
    print(f"{'='*80}\n")
    
    # Configuration
    N_TRAJECTORIES = 50
    N_PREFERENCES = 25
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Chargement de l'agent
    agent_path = os.path.join(results_dir, "mountain_car_agent_classical.pkl")
    
    if not os.path.exists(agent_path):
        print(f"❌ Agent non trouvé: {agent_path}")
        print("💡 Exécutez d'abord: python train_mountaincar_classical.py")
        return
    
    print(f"📂 Chargement de l'agent...")
    agent = MountainCarAgent()
    agent.load_agent(agent_path)
    print("✅ Agent chargé\n")
    
    # Environnement
    print("🌍 Création de l'environnement...")
    env = gym.make('MountainCar-v0')
    print("✅ Environnement créé\n")
    
    # Génération de trajectoires
    print(f"🎬 GÉNÉRATION DE {N_TRAJECTORIES} TRAJECTOIRES")
    print("-" * 80)
    
    trajectories = []
    for i in range(N_TRAJECTORIES):
        traj = collect_mountaincar_trajectory(env, agent, episode_id=i)
        trajectories.append(traj)
        
        if (i + 1) % 10 == 0:
            successes = sum(1 for t in trajectories if t.success)
            print(f"  Trajectoire {i + 1}/{N_TRAJECTORIES} | "
                  f"Succès: {successes}/{i + 1} ({successes/(i+1)*100:.1f}%)")
    
    total_successes = sum(1 for t in trajectories if t.success)
    print(f"\n✅ {N_TRAJECTORIES} trajectoires générées")
    print(f"   Taux de succès: {total_successes}/{N_TRAJECTORIES} ({total_successes/N_TRAJECTORIES*100:.1f}%)\n")
    
    # Sélection de paires
    print(f"📊 SÉLECTION DE {N_PREFERENCES} PAIRES")
    print("-" * 80)
    
    pairs = []
    
    # Tri par performance
    sorted_trajs = sorted(trajectories, key=lambda t: (t.success, t.total_reward), reverse=True)
    
    # Créer des paires variées
    for i in range(0, min(N_PREFERENCES * 2, len(sorted_trajs) - 1), 2):
        if i + 1 < len(sorted_trajs):
            pairs.append((sorted_trajs[i], sorted_trajs[i + 1]))
    
    # Compléter avec paires aléatoires si nécessaire
    while len(pairs) < N_PREFERENCES and len(trajectories) >= 2:
        idx_a = np.random.randint(0, len(trajectories))
        idx_b = np.random.randint(0, len(trajectories))
        if idx_a != idx_b:
            pairs.append((trajectories[idx_a], trajectories[idx_b]))
    
    pairs = pairs[:N_PREFERENCES]
    print(f"✅ {len(pairs)} paires sélectionnées\n")
    
    # Collection automatique
    print(f"🤖 COLLECTE AUTOMATIQUE DE {len(pairs)} PRÉFÉRENCES")
    print("-" * 80)
    
    preferences = []
    
    for idx, (traj_a, traj_b) in enumerate(pairs):
        # Sélection automatique
        choice = auto_select_preference(traj_a, traj_b)
        
        # Affichage
        if (idx + 1) % 5 == 0:
            print(f"  Préférence {idx + 1}/{len(pairs)} collectée")
        
        # Enregistrement
        preference_data = {
            'trajectory_a_id': int(traj_a.episode_id),
            'trajectory_b_id': int(traj_b.episode_id),
            'choice': int(choice),
            'trajectory_a_reward': float(traj_a.total_reward),
            'trajectory_b_reward': float(traj_b.total_reward),
            'trajectory_a_length': int(traj_a.episode_length),
            'trajectory_b_length': int(traj_b.episode_length),
            'trajectory_a_success': bool(traj_a.success),
            'trajectory_b_success': bool(traj_b.success),
            'trajectory_a_max_position': float(traj_a.max_position),
            'trajectory_b_max_position': float(traj_b.max_position),
            'trajectory_a_efficiency': float(traj_a.total_reward / traj_a.episode_length),
            'trajectory_b_efficiency': float(traj_b.total_reward / traj_b.episode_length),
            'timestamp': datetime.now().isoformat(),
            'auto_selected': True
        }
        preferences.append(preference_data)
    
    env.close()
    
    # Sauvegarde
    print(f"\n💾 SAUVEGARDE DES DONNÉES")
    print("-" * 80)
    
    # Préférences
    preferences_path = os.path.join(results_dir, "mountaincar_preferences.json")
    with open(preferences_path, 'w') as f:
        json.dump(preferences, f, indent=2)
    print(f"✅ Préférences sauvegardées: {preferences_path}")
    
    # Trajectoires
    trajectories_path = os.path.join(results_dir, "mountaincar_trajectories.pkl")
    with open(trajectories_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"✅ Trajectoires sauvegardées: {trajectories_path}")
    
    # Statistiques
    print(f"\n{'='*80}")
    print("📊 STATISTIQUES DES PRÉFÉRENCES")
    print(f"{'='*80}")
    print(f"Total préférences: {len(preferences)}")
    
    choices_a = sum(1 for p in preferences if p['choice'] == 1)
    choices_b = sum(1 for p in preferences if p['choice'] == 2)
    choices_equal = sum(1 for p in preferences if p['choice'] == 0)
    
    print(f"  Trajectoire A préférée: {choices_a} ({choices_a/len(preferences)*100:.1f}%)")
    print(f"  Trajectoire B préférée: {choices_b} ({choices_b/len(preferences)*100:.1f}%)")
    print(f"  Égalité: {choices_equal} ({choices_equal/len(preferences)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    print("✅ Collection automatique terminée!")
    print("   Prochaine étape: python train_mountaincar_pbrl.py\n")


if __name__ == "__main__":
    main()
