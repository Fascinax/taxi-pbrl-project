"""
Script de collecte de préférences pour MountainCar-v0
Génère des trajectoires et collecte les préférences humaines
"""

import gymnasium as gym
import numpy as np
import pickle
import json
import os
from typing import List, Tuple
from datetime import datetime
from src.mountain_car_agent import MountainCarAgent
from src.trajectory_manager import TrajectoryManager, Trajectory, TrajectoryStep
from src.preference_interface import PreferenceInterface


class MountainCarTrajectory(Trajectory):
    """Extension de Trajectory avec métriques spécifiques à MountainCar"""
    def __init__(self, steps, total_reward, episode_length, episode_id):
        # L'ordre correct est: steps, total_reward, episode_length, episode_id
        super().__init__(steps, total_reward, episode_length, episode_id)
        self.max_position = -1.2
        self.max_velocity = 0.0
        self.success = False


def collect_mountaincar_trajectory(env: gym.Env, agent: MountainCarAgent, 
                                   episode_id: int) -> MountainCarTrajectory:
    """
    Collecte une trajectoire complète de MountainCar
    
    Args:
        env: Environnement MountainCar
        agent: Agent entraîné
        episode_id: ID de l'épisode
        
    Returns:
        MountainCarTrajectory object
    """
    state, _ = env.reset()
    steps_list = []
    total_reward = 0
    max_position = state[0]
    max_velocity = 0.0
    
    for step_num in range(200):
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Suivi des métriques
        max_position = max(max_position, next_state[0])
        max_velocity = max(abs(max_velocity), abs(next_state[1]))
        
        # Création du step (on garde les états continus mais comme any)
        step = TrajectoryStep(
            state=0,  # Placeholder pour compatibilité
            action=action,
            reward=float(reward),
            next_state=0,  # Placeholder pour compatibilité
            done=done,
            step_number=step_num
        )
        # On stocke les vrais états continus dans des attributs custom
        step.continuous_state = tuple(state)
        step.continuous_next_state = tuple(next_state)
        steps_list.append(step)
        
        total_reward += float(reward)
        state = next_state
        
        if done:
            break
    
    # Création de la trajectoire (ordre correct: steps, total_reward, episode_length, episode_id)
    trajectory = MountainCarTrajectory(
        steps=steps_list,
        total_reward=total_reward,
        episode_length=len(steps_list),
        episode_id=episode_id
    )
    
    # Ajout de métriques spécifiques à MountainCar
    trajectory.max_position = max_position
    trajectory.max_velocity = max_velocity
    trajectory.success = max_position >= 0.5
    
    return trajectory


def display_mountaincar_comparison(traj_a: MountainCarTrajectory, traj_b: MountainCarTrajectory):
    """
    Affiche une comparaison détaillée de deux trajectoires MountainCar
    
    Args:
        traj_a: Première trajectoire
        traj_b: Deuxième trajectoire
    """
    print(f"\n{'='*80}")
    print("[START] COMPARAISON DES TRAJECTOIRES MOUNTAINCAR")
    print(f"{'='*80}\n")
    
    # Trajectoire A
    print(f"[TRAJ A] TRAJECTOIRE A (Épisode {traj_a.episode_id})")
    print("-" * 80)
    print(f"  Récompense totale:     {traj_a.total_reward:7.2f}")
    print(f"  Nombre de pas:         {traj_a.episode_length:3d}")
    print(f"  Position maximale:     {traj_a.max_position:6.3f} {'[OK] But atteint!' if traj_a.success else '[ERROR] Échec'}")
    print(f"  Vitesse maximale:      {traj_a.max_velocity:6.3f}")
    print(f"  Efficacité (reward/pas): {traj_a.total_reward / traj_a.episode_length:6.3f}")
    
    if traj_a.success:
        print(f"  [SUCCESS] Succès en {traj_a.episode_length} pas!")
    else:
        print(f"  [WARN]  N'a atteint que position {traj_a.max_position:.3f}")
    
    print()
    
    # Trajectoire B
    print(f"[TRAJ B] TRAJECTOIRE B (Épisode {traj_b.episode_id})")
    print("-" * 80)
    print(f"  Récompense totale:     {traj_b.total_reward:7.2f}")
    print(f"  Nombre de pas:         {traj_b.episode_length:3d}")
    print(f"  Position maximale:     {traj_b.max_position:6.3f} {'[OK] But atteint!' if traj_b.success else '[ERROR] Échec'}")
    print(f"  Vitesse maximale:      {traj_b.max_velocity:6.3f}")
    print(f"  Efficacité (reward/pas): {traj_b.total_reward / traj_b.episode_length:6.3f}")
    
    if traj_b.success:
        print(f"  [SUCCESS] Succès en {traj_b.episode_length} pas!")
    else:
        print(f"  [WARN]  N'a atteint que position {traj_b.max_position:.3f}")
    
    print()
    
    # Comparaison
    print(f"[COMPARE]  COMPARAISON")
    print("-" * 80)
    
    reward_diff = traj_a.total_reward - traj_b.total_reward
    length_diff = traj_a.episode_length - traj_b.episode_length
    position_diff = traj_a.max_position - traj_b.max_position
    
    if reward_diff > 0:
        print(f"  Récompense:  A est meilleure (+{reward_diff:.2f})")
    elif reward_diff < 0:
        print(f"  Récompense:  B est meilleure (+{abs(reward_diff):.2f})")
    else:
        print(f"  Récompense:  Égalité")
    
    if length_diff < 0:
        print(f"  Efficacité:  A est plus rapide ({abs(length_diff)} pas de moins)")
    elif length_diff > 0:
        print(f"  Efficacité:  B est plus rapide ({length_diff} pas de moins)")
    else:
        print(f"  Efficacité:  Égalité")
    
    if position_diff > 0:
        print(f"  Progression: A monte plus haut (+{position_diff:.3f})")
    elif position_diff < 0:
        print(f"  Progression: B monte plus haut (+{abs(position_diff):.3f})")
    else:
        print(f"  Progression: Égalité")
    
    print(f"\n{'='*80}\n")


def select_interesting_trajectory_pairs(trajectories: List[MountainCarTrajectory], 
                                       n_pairs: int = 20) -> List[Tuple[MountainCarTrajectory, MountainCarTrajectory]]:
    """
    Sélectionne des paires intéressantes de trajectoires pour comparaison
    
    Args:
        trajectories: Liste de trajectoires
        n_pairs: Nombre de paires à sélectionner
        
    Returns:
        Liste de paires (traj_a, traj_b)
    """
    pairs = []
    
    # Séparer succès et échecs
    successes = [t for t in trajectories if t.success]
    failures = [t for t in trajectories if not t.success]
    
    print(f"\n[SELECT] Sélection de {n_pairs} paires intéressantes...")
    print(f"   Trajectoires réussies: {len(successes)}")
    print(f"   Trajectoires échouées: {len(failures)}")
    
    # Type 1: Succès vs Échec (plus clair)
    n_success_fail = min(n_pairs // 3, len(successes), len(failures))
    for i in range(n_success_fail):
        if i < len(successes) and i < len(failures):
            pairs.append((successes[i], failures[i]))
    
    # Type 2: Deux succès avec efficacités différentes
    if len(successes) >= 2:
        successes_sorted = sorted(successes, key=lambda t: t.episode_length)
        n_success_pairs = min(n_pairs // 3, len(successes_sorted) - 1)
        for i in range(0, n_success_pairs * 2, 2):
            if i + 1 < len(successes_sorted):
                pairs.append((successes_sorted[i], successes_sorted[i + 1]))
    
    # Type 3: Échecs avec progressions différentes
    if len(failures) >= 2:
        failures_sorted = sorted(failures, key=lambda t: t.max_position, reverse=True)
        n_failure_pairs = min(n_pairs // 3, len(failures_sorted) - 1)
        for i in range(0, n_failure_pairs * 2, 2):
            if i + 1 < len(failures_sorted):
                pairs.append((failures_sorted[i], failures_sorted[i + 1]))
    
    # Compléter avec paires aléatoires
    while len(pairs) < n_pairs and len(trajectories) >= 2:
        idx_a = np.random.randint(0, len(trajectories))
        idx_b = np.random.randint(0, len(trajectories))
        if idx_a != idx_b:
            pairs.append((trajectories[idx_a], trajectories[idx_b]))
    
    print(f"[OK] {len(pairs)} paires sélectionnées\n")
    return pairs[:n_pairs]


def main():
    """Script principal de collecte de préférences"""
    
    print(f"\n{'='*80}")
    print("[START] COLLECTE DE PRÉFÉRENCES - MOUNTAINCAR-V0")
    print(f"{'='*80}\n")
    
    # Configuration
    N_TRAJECTORIES = 50  # Générer plus de trajectoires
    N_PREFERENCES = 25   # Collecter 25 préférences
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Chargement de l'agent entraîné
    agent_path = os.path.join(results_dir, "mountain_car_agent_classical.pkl")
    
    if not os.path.exists(agent_path):
        print(f"[ERROR] Agent non trouvé: {agent_path}")
        print("[INFO] Exécutez d'abord: python train_mountaincar_classical.py")
        return
    
    print(f"[LOAD] Chargement de l'agent: {agent_path}")
    agent = MountainCarAgent()
    agent.load_agent(agent_path)
    print("[OK] Agent chargé\n")
    
    # Création de l'environnement
    print("[ENV] Création de l'environnement MountainCar-v0...")
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
    
    # Sélection de paires intéressantes
    pairs = select_interesting_trajectory_pairs(trajectories, n_pairs=N_PREFERENCES)
    
    # Collecte des préférences
    print(f"[USER] COLLECTE DE {len(pairs)} PRÉFÉRENCES")
    print(f"{'='*80}")
    print("Instructions:")
    print("  1 ou A : Préférer la trajectoire A")
    print("  2 ou B : Préférer la trajectoire B")
    print("  0 ou E : Égalité / Indifférent")
    print("  Q      : Quitter")
    print(f"{'='*80}\n")
    
    preferences = []
    
    for idx, (traj_a, traj_b) in enumerate(pairs):
        print(f"\n{'='*80}")
        print(f"COMPARAISON {idx + 1}/{len(pairs)}")
        print(f"{'='*80}")
        
        # Affichage de la comparaison
        display_mountaincar_comparison(traj_a, traj_b)
        
        # Collecte de la préférence
        while True:
            choice = input("Votre choix (1/A, 2/B, 0/E, Q pour quitter): ").strip().upper()
            
            if choice in ['Q', 'QUIT']:
                print("\n[WARN]  Arrêt de la collecte...")
                break
            elif choice in ['1', 'A']:
                preference_choice = 1
                print("[OK] Vous avez choisi: Trajectoire A\n")
                break
            elif choice in ['2', 'B']:
                preference_choice = 2
                print("[OK] Vous avez choisi: Trajectoire B\n")
                break
            elif choice in ['0', 'E']:
                preference_choice = 0
                print("[OK] Vous avez choisi: Égalité\n")
                break
            else:
                print("[ERROR] Choix invalide. Utilisez 1/A, 2/B, 0/E, ou Q")
        
        if choice in ['Q', 'QUIT']:
            break
        
        # Enregistrement de la préférence
        preference_data = {
            'trajectory_a_id': int(traj_a.episode_id),
            'trajectory_b_id': int(traj_b.episode_id),
            'choice': int(preference_choice),
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
            'timestamp': datetime.now().isoformat()
        }
        preferences.append(preference_data)
        
        # Sauvegarde intermédiaire
        if (idx + 1) % 5 == 0:
            temp_path = os.path.join(results_dir, f"mountaincar_preferences_temp.json")
            with open(temp_path, 'w') as f:
                json.dump(preferences, f, indent=2)
            print(f"[SAVE] Sauvegarde intermédiaire: {len(preferences)} préférences")
    
    env.close()
    
    # Sauvegarde finale
    if preferences:
        # Sauvegarde des préférences
        preferences_path = os.path.join(results_dir, "mountaincar_preferences.json")
        with open(preferences_path, 'w') as f:
            json.dump(preferences, f, indent=2)
        print(f"\n[SAVE] Préférences sauvegardées: {preferences_path}")
        
        # Sauvegarde des trajectoires
        trajectories_path = os.path.join(results_dir, "mountaincar_trajectories.pkl")
        with open(trajectories_path, 'wb') as f:
            pickle.dump(trajectories, f)
        print(f"[SAVE] Trajectoires sauvegardées: {trajectories_path}")
        
        # Statistiques
        print(f"\n{'='*80}")
        print("[PLOT] STATISTIQUES DES PRÉFÉRENCES")
        print(f"{'='*80}")
        print(f"Total préférences collectées: {len(preferences)}")
        
        choices_a = sum(1 for p in preferences if p['choice'] == 1)
        choices_b = sum(1 for p in preferences if p['choice'] == 2)
        choices_equal = sum(1 for p in preferences if p['choice'] == 0)
        
        print(f"  Trajectoire A préférée: {choices_a} ({choices_a/len(preferences)*100:.1f}%)")
        print(f"  Trajectoire B préférée: {choices_b} ({choices_b/len(preferences)*100:.1f}%)")
        print(f"  Égalité: {choices_equal} ({choices_equal/len(preferences)*100:.1f}%)")
        print(f"{'='*80}\n")
        
        print("[OK] Prochaine étape:")
        print("   python train_mountaincar_pbrl.py")
    else:
        print("\n[WARN]  Aucune préférence collectée")


if __name__ == "__main__":
    main()
