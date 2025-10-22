"""
Agent PBRL (Preference-based Reinforcement Learning) pour MountainCar-v0
Combine Q-Learning avec discrétisation et apprentissage par préférences
"""

import numpy as np
import gymnasium as gym
from typing import List, Dict, Tuple, Any
from src.mountain_car_agent import MountainCarAgent
from src.trajectory_manager import Trajectory


class MountainCarPbRLAgent(MountainCarAgent):
    """
    Agent PBRL spécialisé pour MountainCar avec gestion de la discrétisation
    Hérite de MountainCarAgent pour la discrétisation et étend avec PBRL
    """
    
    def __init__(self,
                 n_position_bins: int = 20,
                 n_velocity_bins: int = 20,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.999,
                 epsilon_min: float = 0.01,
                 preference_weight: float = 0.5):
        """
        Initialise l'agent PbRL pour MountainCar
        
        Args:
            preference_weight: Poids donné aux signaux de préférence
        """
        super().__init__(
            n_position_bins=n_position_bins,
            n_velocity_bins=n_velocity_bins,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min
        )
        
        self.preference_weight = preference_weight
        self.preference_updates = 0
        self.preference_learning_history = []
        
        print(f"🎯 MountainCarPbRLAgent initialisé:")
        print(f"   - Preference weight: {preference_weight}")
    
    def update_from_preferences(self, 
                               preferred_trajectory: Trajectory,
                               less_preferred_trajectory: Trajectory,
                               preference_strength: float = 1.0):
        """
        Met à jour la Q-table en fonction d'une préférence entre deux trajectoires
        
        Args:
            preferred_trajectory: Trajectoire préférée
            less_preferred_trajectory: Trajectoire moins préférée
            preference_strength: Force de la préférence
        """
        # Calcul du bonus/malus
        reward_bonus = preference_strength * self.preference_weight
        reward_penalty = -preference_strength * self.preference_weight * 0.5
        
        # Mise à jour des trajectoires
        self._update_trajectory_values(preferred_trajectory, reward_bonus, is_preferred=True)
        self._update_trajectory_values(less_preferred_trajectory, reward_penalty, is_preferred=False)
        
        self.preference_updates += 1
        
        # Enregistrement
        self.preference_learning_history.append({
            'preferred_reward': preferred_trajectory.total_reward,
            'less_preferred_reward': less_preferred_trajectory.total_reward,
            'preference_strength': preference_strength,
            'reward_bonus': reward_bonus,
            'reward_penalty': reward_penalty
        })
    
    def _update_trajectory_values(self, trajectory: Trajectory, 
                                 reward_modifier: float,
                                 is_preferred: bool):
        """
        Met à jour les valeurs Q pour toutes les transitions d'une trajectoire
        
        Args:
            trajectory: Trajectoire à mettre à jour
            reward_modifier: Modification de récompense
            is_preferred: True si trajectoire préférée
        """
        for i, step in enumerate(trajectory.steps):
            # Position weight: les premières actions sont plus importantes
            position_weight = 1.0 - (i / len(trajectory.steps)) * 0.3
            final_reward = (step.reward + reward_modifier) * position_weight
            
            # Récupération des états continus stockés dans les attributs custom
            if hasattr(step, 'continuous_state') and hasattr(step, 'continuous_next_state'):
                continuous_state = np.array(step.continuous_state)
                continuous_next_state = np.array(step.continuous_next_state)
            else:
                # Fallback: utiliser state/next_state directement s'ils sont déjà continus
                # (pour compatibilité future)
                if isinstance(step.state, (tuple, list)) and len(step.state) == 2:
                    continuous_state = np.array(step.state)
                    continuous_next_state = np.array(step.next_state)
                else:
                    # Skip ce step si on ne peut pas récupérer les états
                    continue
            
            # Conversion des états continus en discrets
            discrete_state = self.process_state(continuous_state)
            discrete_next_state = self.process_state(continuous_next_state)
            
            # Calcul de la target
            if step.done:
                target = final_reward
            else:
                target = final_reward + self.gamma * np.max(self.q_table[discrete_next_state])
            
            # Mise à jour avec learning rate réduit pour les préférences
            preference_lr = self.lr * 0.5
            self.q_table[discrete_state, step.action] += preference_lr * \
                (target - self.q_table[discrete_state, step.action])
    
    def train_with_preferences(self,
                              env: gym.Env,
                              trajectories: List[Trajectory],
                              preferences: List[Dict[str, Any]],
                              episodes: int = 5000) -> List[float]:
        """
        Entraîne l'agent en combinant exploration et apprentissage par préférences
        
        Args:
            env: Environnement MountainCar
            trajectories: Liste de trajectoires pour les préférences
            preferences: Liste des préférences collectées
            episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Liste des récompenses par épisode
        """
        episode_rewards = []
        
        print(f"\n{'='*80}")
        print(f"🎯 ENTRAÎNEMENT PBRL MOUNTAINCAR")
        print(f"{'='*80}")
        print(f"Épisodes: {episodes}")
        print(f"Préférences: {len(preferences)}")
        print()
        
        # Phase 1: Application des préférences existantes
        print("Phase 1: Application des préférences...")
        self._apply_existing_preferences(trajectories, preferences)
        
        # Phase 2: Entraînement avec Q-table modifiée
        print("Phase 2: Entraînement avec exploration...")
        success_count = 0
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 200
            
            while steps < max_steps:
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Mise à jour Q-table
                self.update_q_table(state, action, float(reward), next_state, done)
                
                state = next_state
                total_reward += float(reward)
                steps += 1
                
                if state[0] >= 0.5:
                    success_count += 1
                
                if done:
                    break
            
            self.decay_epsilon()
            episode_rewards.append(total_reward)
            
            # Affichage progrès
            if (episode + 1) % 500 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                success_rate = (success_count / (episode + 1)) * 100
                print(f"Épisode {episode + 1:5d}/{episodes} | "
                      f"Récompense moy. (100 derniers): {avg_reward:7.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Succès: {success_rate:.1f}%")
        
        self.training_rewards = episode_rewards
        
        print(f"\n{'='*80}")
        print("✅ ENTRAÎNEMENT PBRL TERMINÉ")
        print(f"{'='*80}")
        print(f"Récompense moyenne finale: {np.mean(episode_rewards[-100:]):.2f}")
        print(f"Taux de succès: {(success_count / episodes) * 100:.1f}%")
        print(f"Mises à jour par préférences: {self.preference_updates}")
        print(f"{'='*80}\n")
        
        return episode_rewards
    
    def _apply_existing_preferences(self,
                                   trajectories: List[Trajectory],
                                   preferences: List[Dict[str, Any]]):
        """
        Applique les préférences existantes pour initialiser la Q-table
        """
        # Créer mapping des trajectoires
        traj_dict = {traj.episode_id: traj for traj in trajectories}
        
        applied = 0
        for pref in preferences:
            if pref['choice'] == 0:  # Égalité
                continue
            
            traj_a_id = pref['trajectory_a_id']
            traj_b_id = pref['trajectory_b_id']
            
            if traj_a_id not in traj_dict or traj_b_id not in traj_dict:
                continue
            
            traj_a = traj_dict[traj_a_id]
            traj_b = traj_dict[traj_b_id]
            
            # Déterminer trajectoire préférée
            if pref['choice'] == 1:  # A préférée
                preferred = traj_a
                less_preferred = traj_b
            else:  # B préférée
                preferred = traj_b
                less_preferred = traj_a
            
            # Calculer force de préférence
            reward_diff = abs(preferred.total_reward - less_preferred.total_reward)
            
            # Force adaptative
            strength = 1.0 + min(reward_diff / 50.0, 1.0)
            
            # Appliquer préférence
            self.update_from_preferences(preferred, less_preferred, strength)
            applied += 1
        
        print(f"✅ {applied} préférences appliquées (sur {len(preferences)})")
    
    def get_preference_learning_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'apprentissage par préférences"""
        if not self.preference_learning_history:
            return {"error": "Aucun apprentissage par préférences"}
        
        history = self.preference_learning_history
        
        return {
            'total_preference_updates': self.preference_updates,
            'avg_preference_strength': np.mean([h['preference_strength'] for h in history]),
            'avg_reward_bonus': np.mean([h['reward_bonus'] for h in history]),
            'avg_reward_penalty': np.mean([h['reward_penalty'] for h in history]),
            'preference_weight': self.preference_weight
        }
    
    def save_pbrl_agent(self, filepath: str):
        """Sauvegarde l'agent PbRL avec ses données spécifiques"""
        import os
        import pickle
        
        save_data = {
            'q_table': self.q_table,
            'training_rewards': self.training_rewards,
            'preference_updates': self.preference_updates,
            'preference_learning_history': self.preference_learning_history,
            'discretizer_params': {
                'n_position_bins': self.discretizer.n_position_bins,
                'n_velocity_bins': self.discretizer.n_velocity_bins
            },
            'hyperparameters': {
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'preference_weight': self.preference_weight
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"💾 Agent PbRL sauvegardé: {filepath}")


def test_pbrl_agent():
    """Test de l'agent PbRL MountainCar"""
    print("🧪 TEST DE L'AGENT PBRL MOUNTAINCAR\n")
    
    env = gym.make('MountainCar-v0')
    
    # Création agent
    agent = MountainCarPbRLAgent(
        n_position_bins=20,
        n_velocity_bins=20,
        learning_rate=0.1,
        discount_factor=0.99,
        preference_weight=0.5
    )
    
    # Test entraînement rapide
    print("\n🏃 Entraînement rapide (1000 épisodes sans préférences)...")
    agent.train(env, episodes=1000, verbose=True)
    
    # Évaluation
    print("\n📊 Évaluation...")
    agent.evaluate(env, episodes=50, verbose=True)
    
    env.close()
    print("✅ Test terminé!")


if __name__ == "__main__":
    test_pbrl_agent()
