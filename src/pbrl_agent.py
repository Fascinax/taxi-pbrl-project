import numpy as np
from typing import List, Dict, Tuple, Any
from src.q_learning_agent import QLearningAgent
from src.trajectory_manager import Trajectory, TrajectoryStep
from src.preference_interface import PreferenceInterface
import copy

class PreferenceBasedQLearning(QLearningAgent):
    """
    Agent Q-Learning modifié pour apprendre à partir des préférences humaines
    Implémente une version simplifiée du Preference-based Reinforcement Learning
    """
    
    def __init__(self, n_states: int, n_actions: int, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 preference_weight: float = 0.5):
        """
        Initialise l'agent PbRL
        
        Args:
            preference_weight: Poids donné aux signaux de préférence vs récompenses originales
        """
        super().__init__(n_states, n_actions, learning_rate, discount_factor,
                        epsilon, epsilon_decay, epsilon_min)
        
        self.preference_weight = preference_weight
        self.preference_rewards = {}  # Cache des récompenses calculées à partir des préférences
        self.trajectory_values = {}   # Valeurs apprises pour les trajectoires complètes
        
        # Statistiques pour le suivi
        self.preference_updates = 0
        self.preference_learning_history = []
    
    def update_from_preferences(self, preferred_trajectory: Trajectory, 
                              less_preferred_trajectory: Trajectory,
                              preference_strength: float = 1.0):
        """
        Met à jour la Q-table en fonction d'une préférence entre deux trajectoires
        
        Args:
            preferred_trajectory: Trajectoire préférée
            less_preferred_trajectory: Trajectoire moins préférée  
            preference_strength: Force de la préférence (0.5 = légère, 1.0 = forte, 2.0 = très forte)
        """
        
        # Calcul du bonus/malus basé sur la préférence
        reward_bonus = preference_strength * self.preference_weight
        reward_penalty = -preference_strength * self.preference_weight * 0.5  # Pénalité moins forte
        
        # Mise à jour pour la trajectoire préférée (bonus)
        self._update_trajectory_values(preferred_trajectory, reward_bonus, is_preferred=True)
        
        # Mise à jour pour la trajectoire moins préférée (pénalité)
        self._update_trajectory_values(less_preferred_trajectory, reward_penalty, is_preferred=False)
        
        self.preference_updates += 1
        
        # Enregistrement pour analyse
        self.preference_learning_history.append({
            'preferred_reward': preferred_trajectory.total_reward,
            'less_preferred_reward': less_preferred_trajectory.total_reward,
            'preference_strength': preference_strength,
            'reward_bonus': reward_bonus,
            'reward_penalty': reward_penalty
        })
    
    def _update_trajectory_values(self, trajectory: Trajectory, reward_modifier: float, 
                                is_preferred: bool):
        """
        Met à jour les valeurs Q pour toutes les transitions d'une trajectoire
        
        Args:
            trajectory: Trajectoire à mettre à jour
            reward_modifier: Modification de récompense à appliquer
            is_preferred: True si c'est la trajectoire préférée
        """
        
        # Mise à jour de chaque transition dans la trajectoire
        for i, step in enumerate(trajectory.steps):
            # Récompense modifiée basée sur la préférence
            modified_reward = step.reward + reward_modifier
            
            # Facteur de diminution pour les actions en fin de trajectoire
            # (les dernières actions sont moins importantes)
            position_weight = 1.0 - (i / len(trajectory.steps)) * 0.3
            final_reward = modified_reward * position_weight
            
            # Mise à jour Q-Learning standard avec la récompense modifiée
            if step.done:
                target = final_reward
            else:
                target = final_reward + self.gamma * np.max(self.q_table[step.next_state])
            
            # Mise à jour avec un taux d'apprentissage réduit pour les préférences
            preference_lr = self.lr * 0.5  # Apprentissage plus conservateur
            self.q_table[step.state, step.action] += preference_lr * \
                (target - self.q_table[step.state, step.action])
    
    def train_with_preferences(self, env, trajectories: List[Trajectory], 
                             preferences: List[Dict[str, Any]], 
                             episodes: int = 5000) -> List[float]:
        """
        Entraîne l'agent en combinant exploration normale et apprentissage par préférences
        
        Args:
            env: Environnement Gymnasium
            trajectories: Liste des trajectoires pour les préférences
            preferences: Liste des préférences collectées
            episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Liste des récompenses par épisode
        """
        episode_rewards = []
        
        print(f"Entraînement PbRL: {episodes} épisodes avec {len(preferences)} préférences")
        
        # Phase 1: Apprentissage initial par préférences
        print("Phase 1: Application des préférences existantes...")
        self._apply_existing_preferences(trajectories, preferences)
        
        # Phase 2: Entraînement normal avec Q-table modifiée
        print("Phase 2: Entraînement avec exploration...")
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 200
            
            while steps < max_steps:
                # Sélection et exécution de l'action
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Mise à jour normale de la Q-table
                self.update_q_table(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Mise à jour d'epsilon
            self.decay_epsilon()
            
            episode_rewards.append(total_reward)
            
            # Affichage du progrès
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Épisode {episode + 1}/{episodes}, "
                      f"Récompense moyenne (100 derniers): {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        self.training_rewards = episode_rewards
        return episode_rewards
    
    def _apply_existing_preferences(self, trajectories: List[Trajectory], 
                                  preferences: List[Dict[str, Any]]):
        """
        Applique les préférences existantes pour initialiser la Q-table
        """
        
        # Créer un mapping des trajectoires par ID
        traj_dict = {traj.episode_id: traj for traj in trajectories}
        
        for pref in preferences:
            if pref['choice'] == 0:  # Égalité, on ignore
                continue
            
            traj_a_id = pref['trajectory_a_id']
            traj_b_id = pref['trajectory_b_id']
            
            if traj_a_id not in traj_dict or traj_b_id not in traj_dict:
                continue
            
            traj_a = traj_dict[traj_a_id]
            traj_b = traj_dict[traj_b_id]
            
            # Déterminer quelle trajectoire est préférée
            if pref['choice'] == 1:  # Trajectoire A préférée
                preferred = traj_a
                less_preferred = traj_b
            else:  # Trajectoire B préférée
                preferred = traj_b
                less_preferred = traj_a
            
            # Calculer la force de préférence basée sur la différence de performance
            reward_diff = abs(preferred.total_reward - less_preferred.total_reward)
            efficiency_diff = abs(pref['trajectory_a_efficiency'] - pref['trajectory_b_efficiency'])
            
            # Force adaptative basée sur les différences
            strength = 1.0 + min(reward_diff / 10.0, 1.0) + min(efficiency_diff, 0.5)
            
            # Appliquer la préférence
            self.update_from_preferences(preferred, less_preferred, strength)
        
        print(f"✅ {len([p for p in preferences if p['choice'] != 0])} préférences appliquées")
    
    def interactive_training_loop(self, env, preference_interface: PreferenceInterface,
                                trajectory_manager, episodes_per_iteration: int = 1000,
                                max_iterations: int = 5, trajectories_per_comparison: int = 5):
        """
        Boucle d'entraînement interactive avec collecte de préférences en temps réel
        
        Args:
            env: Environnement
            preference_interface: Interface de collecte de préférences
            trajectory_manager: Gestionnaire de trajectoires
            episodes_per_iteration: Épisodes entre chaque collecte de préférences
            max_iterations: Nombre maximum d'itérations
            trajectories_per_comparison: Nombre de trajectoires à générer pour comparaison
        """
        
        print(f"🚀 ENTRAÎNEMENT INTERACTIF PbRL")
        print(f"Paramètres: {max_iterations} itérations, {episodes_per_iteration} épisodes/itération")
        
        all_rewards = []
        iteration_summaries = []
        
        for iteration in range(max_iterations):
            print(f"\n🔄 ITÉRATION {iteration + 1}/{max_iterations}")
            
            # 1. Entraînement standard
            print(f"1️⃣ Entraînement standard ({episodes_per_iteration} épisodes)...")
            iteration_rewards = self.train(env, episodes=episodes_per_iteration)
            all_rewards.extend(iteration_rewards)
            
            # 2. Génération de trajectoires pour comparaison
            print(f"2️⃣ Génération de {trajectories_per_comparison} trajectoires de test...")
            test_trajectories = []
            for i in range(trajectories_per_comparison):
                traj = trajectory_manager.collect_trajectory(env, self, render=False)
                test_trajectories.append(traj)
            
            # 3. Sélection de paires intéressantes
            print("3️⃣ Sélection de paires pour comparaison...")
            pairs = self._select_interesting_pairs(test_trajectories)
            
            if not pairs:
                print("⚠️ Aucune paire intéressante trouvée, passage à l'itération suivante")
                continue
            
            # 4. Collecte de préférences
            print(f"4️⃣ Collecte de préférences ({len(pairs)} comparaisons)...")
            preferences = preference_interface.collect_preference_batch(
                pairs, trajectory_manager, 
                env=env, 
                visual_replay=True
            )
            
            # 5. Application des nouvelles préférences
            print("5️⃣ Application des nouvelles préférences...")
            self._apply_new_preferences(pairs, preferences)
            
            # 6. Résumé de l'itération
            iteration_summary = {
                'iteration': iteration + 1,
                'avg_reward': np.mean(iteration_rewards),
                'preferences_collected': len([p for p in preferences if p != 0]),
                'total_preference_updates': self.preference_updates
            }
            iteration_summaries.append(iteration_summary)
            
            print(f"📊 Résumé itération {iteration + 1}:")
            print(f"   Récompense moyenne: {iteration_summary['avg_reward']:.2f}")
            print(f"   Préférences collectées: {iteration_summary['preferences_collected']}")
            
        # Résumé final
        print(f"\n🎉 ENTRAÎNEMENT INTERACTIF TERMINÉ")
        print(f"Total des mises à jour par préférences: {self.preference_updates}")
        print(f"Récompense finale moyenne: {np.mean(all_rewards[-100:]):.2f}")
        
        return all_rewards, iteration_summaries
    
    def _select_interesting_pairs(self, trajectories: List[Trajectory]) -> List[Tuple[Trajectory, Trajectory]]:
        """Sélectionne des paires intéressantes pour la comparaison"""
        if len(trajectories) < 2:
            return []
        
        # Tri par performance pour créer des contrastes
        sorted_trajs = sorted(trajectories, key=lambda t: t.total_reward, reverse=True)
        
        pairs = []
        
        # Paire best vs worst si différence significative
        if len(sorted_trajs) >= 2:
            best = sorted_trajs[0]
            worst = sorted_trajs[-1]
            if abs(best.total_reward - worst.total_reward) > 2:  # Différence significative
                pairs.append((best, worst))
        
        # Paire avec récompenses similaires mais efficacités différentes
        middle_idx = len(sorted_trajs) // 2
        if middle_idx > 0 and middle_idx < len(sorted_trajs) - 1:
            traj1 = sorted_trajs[middle_idx]
            traj2 = sorted_trajs[middle_idx + 1]
            # Vérifier si différence d'efficacité significative
            eff1 = traj1.total_reward / traj1.episode_length
            eff2 = traj2.total_reward / traj2.episode_length
            if abs(eff1 - eff2) > 0.1:
                pairs.append((traj1, traj2))
        
        return pairs
    
    def _apply_new_preferences(self, pairs: List[Tuple[Trajectory, Trajectory]], 
                             preferences: List[int]):
        """Applique les nouvelles préférences collectées"""
        
        for (traj1, traj2), preference in zip(pairs, preferences):
            if preference == 0:  # Égalité
                continue
            elif preference == 1:  # Préfère traj1
                self.update_from_preferences(traj1, traj2, preference_strength=1.0)
            elif preference == 2:  # Préfère traj2
                self.update_from_preferences(traj2, traj1, preference_strength=1.0)
    
    def get_preference_learning_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'apprentissage par préférences"""
        
        if not self.preference_learning_history:
            return {"error": "Aucun apprentissage par préférences effectué"}
        
        history = self.preference_learning_history
        
        return {
            'total_preference_updates': self.preference_updates,
            'avg_preference_strength': np.mean([h['preference_strength'] for h in history]),
            'avg_reward_bonus': np.mean([h['reward_bonus'] for h in history]),
            'avg_reward_penalty': np.mean([h['reward_penalty'] for h in history]),
            'reward_differences': [h['preferred_reward'] - h['less_preferred_reward'] for h in history],
            'preference_weight_used': self.preference_weight
        }
    
    def save_pbrl_agent(self, filepath: str):
        """Sauvegarde l'agent PbRL avec ses données spécifiques"""
        save_data = {
            'q_table': self.q_table,
            'training_rewards': self.training_rewards,
            'preference_updates': self.preference_updates,
            'preference_learning_history': self.preference_learning_history,
            'preference_weight': self.preference_weight,
            'hyperparameters': {
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'preference_weight': self.preference_weight
            }
        }
        
        import os
        import pickle
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Agent PbRL sauvegardé: {filepath}")