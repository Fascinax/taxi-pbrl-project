"""
Agent Q-Learning adapté pour MountainCar-v0 avec discrétisation d'états
"""

import numpy as np
import gymnasium as gym
from typing import List, Tuple, Optional
from src.q_learning_agent import QLearningAgent
from src.mountain_car_discretizer import MountainCarDiscretizer


class MountainCarAgent(QLearningAgent):
    """
    Agent Q-Learning spécialisé pour MountainCar avec discrétisation d'états continus
    """
    
    def __init__(self, 
                 n_position_bins: int = 20,
                 n_velocity_bins: int = 20,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.999,
                 epsilon_min: float = 0.01):
        """
        Initialise l'agent MountainCar
        
        Args:
            n_position_bins: Nombre de bins pour discrétiser la position
            n_velocity_bins: Nombre de bins pour discrétiser la vitesse
        """
        
        # Initialisation du discrétiseur
        self.discretizer = MountainCarDiscretizer(n_position_bins, n_velocity_bins)
        
        # Initialisation de l'agent Q-Learning parent
        n_states = self.discretizer.n_states
        n_actions = 3  # MountainCar: 0=gauche, 1=rien, 2=droite
        
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min
        )
        
        print(f"🚗 MountainCarAgent initialisé:")
        print(f"   - États discrets: {n_states}")
        print(f"   - Actions: {n_actions} (0=gauche, 1=rien, 2=droite)")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Gamma: {discount_factor}")
    
    def process_state(self, continuous_state: np.ndarray) -> int:
        """
        Convertit un état continu en état discret
        
        Args:
            continuous_state: État [position, vitesse]
            
        Returns:
            État discret (index)
        """
        return self.discretizer.discretize(continuous_state)
    
    def select_action(self, continuous_state: np.ndarray, training: bool = True) -> int:
        """
        Sélectionne une action pour un état continu
        
        Args:
            continuous_state: État [position, vitesse]
            training: Si True, utilise epsilon-greedy
            
        Returns:
            Action sélectionnée
        """
        discrete_state = self.process_state(continuous_state)
        return super().select_action(discrete_state, training)
    
    def update_q_table(self, continuous_state: np.ndarray, action: int, 
                      reward: float, continuous_next_state: np.ndarray, done: bool):
        """
        Met à jour la Q-table avec des états continus
        
        Args:
            continuous_state: État actuel [position, vitesse]
            action: Action prise
            reward: Récompense reçue
            continuous_next_state: État suivant [position, vitesse]
            done: Épisode terminé
        """
        state = self.process_state(continuous_state)
        next_state = self.process_state(continuous_next_state)
        super().update_q_table(state, action, reward, next_state, done)
    
    def train(self, env: gym.Env, episodes: int = 10000, 
             max_steps: int = 200, verbose: bool = True) -> List[float]:
        """
        Entraîne l'agent sur MountainCar
        
        Args:
            env: Environnement MountainCar
            episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximum de pas par épisode
            verbose: Afficher les progrès
            
        Returns:
            Liste des récompenses par épisode
        """
        episode_rewards = []
        success_count = 0
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"[START] ENTRAÎNEMENT MOUNTAINCAR - {episodes} épisodes")
            print(f"{'='*80}\n")
        
        for episode in range(episodes):
            continuous_state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Sélection et exécution de l'action
                action = self.select_action(continuous_state, training=True)
                next_continuous_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Mise à jour de la Q-table
                self.update_q_table(continuous_state, action, reward, 
                                   next_continuous_state, done)
                
                continuous_state = next_continuous_state
                total_reward += reward
                steps += 1
                
                # Vérification de succès
                if continuous_state[0] >= 0.5:  # Position >= 0.5 = but atteint
                    success_count += 1
                
                if done:
                    break
            
            # Diminution d'epsilon
            self.decay_epsilon()
            
            episode_rewards.append(total_reward)
            
            # Affichage des progrès
            if verbose and (episode + 1) % 500 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                success_rate = (success_count / (episode + 1)) * 100
                print(f"Épisode {episode + 1:5d}/{episodes} | "
                      f"Récompense moy. (100 derniers): {avg_reward:7.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Succès: {success_rate:.1f}%")
        
        self.training_rewards = episode_rewards
        
        if verbose:
            final_success_rate = (success_count / episodes) * 100
            print(f"\n{'='*80}")
            print(f"[OK] ENTRAÎNEMENT TERMINÉ")
            print(f"{'='*80}")
            print(f"Récompense moyenne finale (100 derniers): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"Taux de succès: {final_success_rate:.1f}%")
            print(f"Epsilon final: {self.epsilon:.3f}")
            print(f"{'='*80}\n")
        
        return episode_rewards
    
    def evaluate(self, env: gym.Env, episodes: int = 100, 
                render: bool = False, verbose: bool = True) -> Tuple[List[float], dict]:
        """
        Évalue l'agent sur MountainCar
        
        Args:
            env: Environnement MountainCar
            episodes: Nombre d'épisodes d'évaluation
            render: Afficher les épisodes
            verbose: Afficher les résultats
            
        Returns:
            Tuple (liste de récompenses, statistiques)
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(episodes):
            continuous_state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 200:
                action = self.select_action(continuous_state, training=False)
                continuous_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                if continuous_state[0] >= 0.5:
                    success_count += 1
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        # Calcul des statistiques
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': (success_count / episodes) * 100,
            'total_episodes': episodes
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"[PLOT] ÉVALUATION MOUNTAINCAR - {episodes} épisodes")
            print(f"{'='*80}")
            print(f"Récompense moyenne: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"Récompense min/max: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
            print(f"Longueur moyenne: {stats['mean_length']:.1f} ± {stats['std_length']:.1f} pas")
            print(f"Taux de succès: {stats['success_rate']:.1f}%")
            print(f"{'='*80}\n")
        
        return episode_rewards, stats
    
    def get_action_name(self, action: int) -> str:
        """Retourne le nom d'une action"""
        action_names = {
            0: "← Gauche",
            1: "⊙ Rien",
            2: "→ Droite"
        }
        return action_names.get(action, f"Action {action}")
    
    def get_state_analysis(self, continuous_state: np.ndarray) -> dict:
        """
        Analyse un état et retourne des informations détaillées
        
        Args:
            continuous_state: État [position, vitesse]
            
        Returns:
            Dictionnaire d'analyses
        """
        discrete_state = self.process_state(continuous_state)
        state_info = self.discretizer.get_state_info(continuous_state)
        
        # Valeurs Q pour cet état
        q_values = self.q_table[discrete_state]
        best_action = np.argmax(q_values)
        
        return {
            **state_info,
            'discrete_state': discrete_state,
            'q_values': q_values,
            'best_action': best_action,
            'best_action_name': self.get_action_name(best_action),
            'q_value_best': q_values[best_action]
        }
    
    def visualize_policy(self, sample_states: Optional[List[Tuple[float, float]]] = None):
        """
        Visualise la politique apprise pour quelques états
        
        Args:
            sample_states: Liste d'états à analyser (si None, utilise des états par défaut)
        """
        if sample_states is None:
            # États par défaut représentatifs
            sample_states = [
                (-1.2, 0.0),   # Départ
                (-0.8, -0.05), # Vallée gauche avec vitesse gauche
                (-0.5, 0.0),   # Centre vallée
                (-0.2, 0.05),  # Montée droite avec élan
                (0.0, 0.0),    # Milieu
                (0.3, 0.05),   # Proche du but
            ]
        
        print(f"\n{'='*80}")
        print("[MAP]  POLITIQUE APPRISE (échantillon d'états)")
        print(f"{'='*80}\n")
        
        for position, velocity in sample_states:
            state = np.array([position, velocity])
            analysis = self.get_state_analysis(state)
            
            print(f"Position: {position:5.2f} | Vitesse: {velocity:6.3f}")
            print(f"  → Action recommandée: {analysis['best_action_name']}")
            print(f"  → Valeur Q: {analysis['q_value_best']:.2f}")
            print(f"  → Progrès: {analysis['progress_percent']:.1f}%")
            print()
        
        print(f"{'='*80}\n")


def test_mountain_car_agent():
    """Test de l'agent MountainCar"""
    print("🧪 TEST DE L'AGENT MOUNTAINCAR\n")
    
    # Création de l'environnement
    env = gym.make('MountainCar-v0')
    
    # Création de l'agent
    agent = MountainCarAgent(
        n_position_bins=20,
        n_velocity_bins=20,
        learning_rate=0.1,
        discount_factor=0.99
    )
    
    # Test rapide d'entraînement
    print("\n🏃 Entraînement rapide (1000 épisodes)...")
    agent.train(env, episodes=1000, verbose=True)
    
    # Évaluation
    print("\n[PLOT] Évaluation...")
    agent.evaluate(env, episodes=50, verbose=True)
    
    # Visualisation de la politique
    agent.visualize_policy()
    
    env.close()
    print("[OK] Test terminé!")


if __name__ == "__main__":
    test_mountain_car_agent()
