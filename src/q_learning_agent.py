import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import pickle
import os

class QLearningAgent:
    """
    Agent Q-Learning classique pour l'environnement Taxi-v3
    """
    
    def __init__(self, n_states: int, n_actions: int, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialise l'agent Q-Learning
        
        Args:
            n_states: Nombre d'états dans l'environnement
            n_actions: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage (alpha)
            discount_factor: Facteur de réduction (gamma)
            epsilon: Taux d'exploration initial
            epsilon_decay: Facteur de décroissance d'epsilon
            epsilon_min: Valeur minimale d'epsilon
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialisation de la Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Métriques pour le suivi
        self.training_rewards = []
        self.training_episodes = []
        self.epsilons = []
        
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Sélectionne une action selon la politique epsilon-greedy
        
        Args:
            state: État actuel
            training: Si True, utilise epsilon-greedy, sinon greedy
            
        Returns:
            Action à prendre
        """
        if training and np.random.random() < self.epsilon:
            # Exploration : action aléatoire
            return np.random.randint(0, self.n_actions)
        else:
            # Exploitation : meilleure action selon Q-table
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool):
        """
        Met à jour la Q-table avec la règle de mise à jour Q-Learning
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Si l'épisode est terminé
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Mise à jour Q-Learning
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
    
    def decay_epsilon(self):
        """Réduit epsilon après chaque épisode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, episodes: int = 10000, max_steps: int = 200) -> List[float]:
        """
        Entraîne l'agent sur l'environnement
        
        Args:
            env: Environnement Gymnasium
            episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximum de pas par épisode
            
        Returns:
            Liste des récompenses par épisode
        """
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < max_steps:
                # Sélection et exécution de l'action
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Mise à jour de la Q-table
                self.update_q_table(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Mise à jour d'epsilon
            self.decay_epsilon()
            
            # Sauvegarde des métriques
            episode_rewards.append(total_reward)
            
            # Affichage du progrès
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Épisode {episode + 1}/{episodes}, "
                      f"Récompense moyenne (100 derniers): {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        self.training_rewards = episode_rewards
        return episode_rewards
    
    def evaluate(self, env, episodes: int = 100, max_steps: int = 200, 
                render: bool = False) -> Tuple[float, List[float]]:
        """
        Évalue la performance de l'agent entraîné
        
        Args:
            env: Environnement Gymnasium
            episodes: Nombre d'épisodes d'évaluation
            max_steps: Nombre maximum de pas par épisode
            render: Si True, affiche l'environnement
            
        Returns:
            Tuple (récompense moyenne, liste des récompenses)
        """
        evaluation_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < max_steps:
                action = self.select_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                if render and episode < 5:  # Affiche seulement les 5 premiers épisodes
                    env.render()
                
                if done:
                    break
            
            evaluation_rewards.append(total_reward)
        
        avg_reward = np.mean(evaluation_rewards)
        print(f"Évaluation sur {episodes} épisodes:")
        print(f"Récompense moyenne: {avg_reward:.2f}")
        print(f"Récompense médiane: {np.median(evaluation_rewards):.2f}")
        print(f"Écart-type: {np.std(evaluation_rewards):.2f}")
        
        return avg_reward, evaluation_rewards
    
    def plot_training_progress(self, save_path: str = None):
        """
        Trace les courbes d'apprentissage
        
        Args:
            save_path: Chemin pour sauvegarder le graphique
        """
        if not self.training_rewards:
            print("Aucune donnée d'entraînement disponible")
            return
        
        # Calcul de la moyenne mobile
        window_size = 100
        moving_avg = []
        for i in range(len(self.training_rewards)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(self.training_rewards[start_idx:i+1]))
        
        plt.figure(figsize=(12, 5))
        
        # Graphique des récompenses
        plt.subplot(1, 2, 1)
        plt.plot(self.training_rewards, alpha=0.3, color='blue', label='Récompense par épisode')
        plt.plot(moving_avg, color='red', label=f'Moyenne mobile ({window_size} épisodes)')
        plt.xlabel('Épisode')
        plt.ylabel('Récompense')
        plt.title('Progression de l\'entraînement')
        plt.legend()
        plt.grid(True)
        
        # Graphique d'epsilon
        plt.subplot(1, 2, 2)
        epsilons = [self.epsilon_decay ** i for i in range(len(self.training_rewards))]
        plt.plot(epsilons)
        plt.xlabel('Épisode')
        plt.ylabel('Epsilon')
        plt.title('Décroissance d\'Epsilon')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def save_agent(self, filepath: str):
        """Sauvegarde l'agent entraîné"""
        save_data = {
            'q_table': self.q_table,
            'training_rewards': self.training_rewards,
            'hyperparameters': {
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Agent sauvegardé: {filepath}")
    
    def load_agent(self, filepath: str):
        """Charge un agent sauvegardé"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_table = save_data['q_table']
        self.training_rewards = save_data.get('training_rewards', [])
        
        print(f"Agent chargé: {filepath}")