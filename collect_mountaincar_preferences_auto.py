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
    Sélection automatique OPTIMISÉE basée sur des critères objectifs hiérarchiques
    
    Returns:
        1 si A est meilleure, 2 si B est meilleure, 0 si égalité
    """
    # Critère 1 (Priorité MAX): Succès vs échec
    if traj_a.success and not traj_b.success:
        return 1
    if traj_b.success and not traj_a.success:
        return 2
    
    # Critère 2: Si les deux réussissent, préférer la plus rapide (efficacité)
    if traj_a.success and traj_b.success:
        length_diff = abs(traj_a.episode_length - traj_b.episode_length)
        if length_diff > 3:  # Seuil abaissé pour être plus discriminant
            return 1 if traj_a.episode_length < traj_b.episode_length else 2
    
    # Critère 3: Si les deux échouent, préférer celle qui monte BEAUCOUP plus haut
    if not traj_a.success and not traj_b.success:
        position_diff = abs(traj_a.max_position - traj_b.max_position)
        
        # Préférence forte si différence significative
        if position_diff > 0.1:  # Différence marquée
            return 1 if traj_a.max_position > traj_b.max_position else 2
        elif position_diff > 0.03:  # Différence modérée
            # Double-check avec la récompense
            if abs(traj_a.total_reward - traj_b.total_reward) > 3:
                return 1 if traj_a.total_reward > traj_b.total_reward else 2
    
    # Critère 4: Différence de récompense significative
    reward_diff = abs(traj_a.total_reward - traj_b.total_reward)
    if reward_diff > 10:  # Seuil augmenté pour éviter les égalités
        return 1 if traj_a.total_reward > traj_b.total_reward else 2
    
    # Critère 5: Efficacité (récompense / longueur)
    efficiency_a = traj_a.total_reward / max(traj_a.episode_length, 1)
    efficiency_b = traj_b.total_reward / max(traj_b.episode_length, 1)
    efficiency_diff = abs(efficiency_a - efficiency_b)
    
    if efficiency_diff > 0.15:
        return 1 if efficiency_a > efficiency_b else 2
    
    # Sinon, égalité (rare maintenant)
    return 0


def main():
    """Collection automatique de préférences"""
    
    print(f"\n{'='*80}")
    print("[AGENT] COLLECTE AUTOMATIQUE DE PRÉFÉRENCES - MOUNTAINCAR-V0")
    print(f"{'='*80}\n")
    
    # Configuration
    N_TRAJECTORIES = 80  # Augmenté pour plus de diversité
    N_PREFERENCES = 40   # Augmenté pour plus d'apprentissage
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Chargement de l'agent
    agent_path = os.path.join(results_dir, "mountain_car_agent_classical.pkl")
    
    if not os.path.exists(agent_path):
        print(f"[ERROR] Agent non trouvé: {agent_path}")
        print("[INFO] Exécutez d'abord: python train_mountaincar_classical.py")
        return
    
    print(f"[LOAD] Chargement de l'agent...")
    agent = MountainCarAgent()
    agent.load_agent(agent_path)
    print("[OK] Agent chargé\n")
    
    # Environnement
    print("[ENV] Création de l'environnement...")
    env = gym.make('MountainCar-v0')
    print("[OK] Environnement créé\n")
    
    # Génération de trajectoires
    print(f"[ACTION] GÉNÉRATION DE {N_TRAJECTORIES} TRAJECTOIRES")
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
    print(f"\n[OK] {N_TRAJECTORIES} trajectoires générées")
    print(f"   Taux de succès: {total_successes}/{N_TRAJECTORIES} ({total_successes/N_TRAJECTORIES*100:.1f}%)\n")
    
    # Sélection de paires OPTIMISÉE pour contraste maximum
    print(f"[PLOT] SÉLECTION DE {N_PREFERENCES} PAIRES")
    print("-" * 80)
    
    pairs = []
    
    # Tri par performance
    sorted_trajs = sorted(trajectories, key=lambda t: (t.success, t.total_reward, -t.episode_length), reverse=True)
    
    # Séparer succès et échecs
    success_trajs = [t for t in sorted_trajs if t.success]
    fail_trajs = [t for t in sorted_trajs if not t.success]
    
    print(f"   Succès: {len(success_trajs)}, Échecs: {len(fail_trajs)}")
    
    # Stratégie 1: Paires succès vs échec (contraste maximum)
    min_contrasted = min(len(success_trajs), len(fail_trajs), N_PREFERENCES // 2)
    for i in range(min_contrasted):
        pairs.append((success_trajs[i], fail_trajs[i]))
    
    # Stratégie 2: Paires au sein des succès (meilleur vs moins bon)
    if len(success_trajs) >= 4:
        for i in range(0, min(len(success_trajs) // 2, N_PREFERENCES // 4)):
            if i * 2 + 1 < len(success_trajs):
                pairs.append((success_trajs[i], success_trajs[len(success_trajs) - 1 - i]))
    
    # Stratégie 3: Paires au sein des échecs (haut vs bas)
    if len(fail_trajs) >= 4:
        fail_sorted = sorted(fail_trajs, key=lambda t: t.max_position, reverse=True)
        for i in range(0, min(len(fail_sorted) // 2, N_PREFERENCES // 4)):
            if i * 2 + 1 < len(fail_sorted):
                pairs.append((fail_sorted[i], fail_sorted[len(fail_sorted) - 1 - i]))
    
    # Compléter avec paires adjacentes variées si nécessaire
    remaining = N_PREFERENCES - len(pairs)
    if remaining > 0 and len(sorted_trajs) >= 2:
        step = max(2, len(sorted_trajs) // (remaining + 1))
        for i in range(0, len(sorted_trajs) - step, step):
            if len(pairs) >= N_PREFERENCES:
                break
            pairs.append((sorted_trajs[i], sorted_trajs[i + step]))
    
    pairs = pairs[:N_PREFERENCES]
    print(f"[OK] {len(pairs)} paires sélectionnées\n")
    
    # Collection automatique
    print(f"[AGENT] COLLECTE AUTOMATIQUE DE {len(pairs)} PRÉFÉRENCES")
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
    print(f"\n[SAVE] SAUVEGARDE DES DONNÉES")
    print("-" * 80)
    
    # Préférences
    preferences_path = os.path.join(results_dir, "mountaincar_preferences.json")
    with open(preferences_path, 'w') as f:
        json.dump(preferences, f, indent=2)
    print(f"[OK] Préférences sauvegardées: {preferences_path}")
    
    # Trajectoires
    trajectories_path = os.path.join(results_dir, "mountaincar_trajectories.pkl")
    with open(trajectories_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"[OK] Trajectoires sauvegardées: {trajectories_path}")
    
    # Statistiques
    print(f"\n{'='*80}")
    print("[PLOT] STATISTIQUES DES PRÉFÉRENCES")
    print(f"{'='*80}")
    print(f"Total préférences: {len(preferences)}")
    
    choices_a = sum(1 for p in preferences if p['choice'] == 1)
    choices_b = sum(1 for p in preferences if p['choice'] == 2)
    choices_equal = sum(1 for p in preferences if p['choice'] == 0)
    
    print(f"  Trajectoire A préférée: {choices_a} ({choices_a/len(preferences)*100:.1f}%)")
    print(f"  Trajectoire B préférée: {choices_b} ({choices_b/len(preferences)*100:.1f}%)")
    print(f"  Égalité: {choices_equal} ({choices_equal/len(preferences)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    print("[OK] Collection automatique terminée!")
    print("   Prochaine étape: python train_mountaincar_pbrl.py\n")


if __name__ == "__main__":
    main()
