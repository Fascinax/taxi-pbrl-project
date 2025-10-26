import numpy as np
from typing import List, Tuple, Dict, Any
import pickle
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import gymnasium as gym

@dataclass
class TrajectoryStep:
    """Représente un pas dans une trajectoire"""
    state: int
    action: int
    reward: float
    next_state: int
    done: bool
    step_number: int

@dataclass
class Trajectory:
    """Représente une trajectoire complète (épisode)"""
    steps: List[TrajectoryStep]
    total_reward: float
    episode_length: int
    episode_id: int
    
    def __post_init__(self):
        if not self.steps:
            self.total_reward = 0
            self.episode_length = 0
        else:
            self.total_reward = sum(step.reward for step in self.steps)
            self.episode_length = len(self.steps)

class TrajectoryManager:
    """
    Gère la collecte, le stockage et la comparaison de trajectoires
    """
    
    def __init__(self):
        self.trajectories: List[Trajectory] = []
        self.trajectory_counter = 0
        
    def collect_trajectory(self, env, agent, max_steps: int = 200, 
                          render: bool = False) -> Trajectory:
        """
        Collecte une trajectoire complète en faisant jouer l'agent
        
        Args:
            env: Environnement Gymnasium
            agent: Agent Q-Learning
            max_steps: Nombre maximum de pas
            render: Si True, affiche l'environnement
            
        Returns:
            Trajectory: Trajectoire complète
        """
        steps = []
        state, _ = env.reset()
        total_reward = 0
        step_number = 0
        
        while step_number < max_steps:
            # Agent choisit une action
            action = agent.select_action(state, training=False)
            
            # Exécution de l'action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Enregistrement du pas
            step = TrajectoryStep(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                step_number=step_number
            )
            steps.append(step)
            
            total_reward += reward
            state = next_state
            step_number += 1
            
            if render:
                env.render()
            
            if done:
                break
        
        # Création de la trajectoire
        trajectory = Trajectory(
            steps=steps,
            total_reward=total_reward,
            episode_length=len(steps),
            episode_id=self.trajectory_counter
        )
        
        self.trajectory_counter += 1
        self.trajectories.append(trajectory)
        
        return trajectory
    
    def get_trajectory_summary(self, trajectory: Trajectory) -> Dict[str, Any]:
        """
        Génère un résumé lisible d'une trajectoire
        
        Args:
            trajectory: Trajectoire à résumer
            
        Returns:
            Dict contenant le résumé
        """
        actions_names = ["Sud", "Nord", "Est", "Ouest", "Prendre", "Déposer"]
        
        # Statistiques de base
        summary = {
            'episode_id': trajectory.episode_id,
            'total_reward': trajectory.total_reward,
            'episode_length': trajectory.episode_length,
            'actions_taken': [actions_names[step.action] for step in trajectory.steps],
            'action_counts': {},
            'rewards_distribution': {},
            'success': trajectory.total_reward > 0  # Heuristique: >0 = succès
        }
        
        # Comptage des actions
        for action_name in actions_names:
            summary['action_counts'][action_name] = summary['actions_taken'].count(action_name)
        
        # Distribution des récompenses
        rewards = [step.reward for step in trajectory.steps]
        unique_rewards = list(set(rewards))
        for reward in unique_rewards:
            summary['rewards_distribution'][reward] = rewards.count(reward)
        
        # Efficacité (récompense par pas)
        summary['efficiency'] = trajectory.total_reward / trajectory.episode_length if trajectory.episode_length > 0 else 0
        
        return summary
    
    def display_trajectory_comparison(self, traj1: Trajectory, traj2: Trajectory):
        """
        Affiche une comparaison détaillée entre deux trajectoires
        
        Args:
            traj1: Première trajectoire
            traj2: Deuxième trajectoire
        """
        summary1 = self.get_trajectory_summary(traj1)
        summary2 = self.get_trajectory_summary(traj2)
        
        print("\n" + "="*80)
        print("COMPARAISON DE TRAJECTOIRES")
        print("="*80)
        
        # Comparaison générale
        print(f"\n📊 STATISTIQUES GÉNÉRALES:")
        print(f"{'Métrique':<20} {'Trajectoire A':<15} {'Trajectoire B':<15} {'Meilleure':<10}")
        print("-" * 65)
        
        metrics = [
            ("Récompense totale", summary1['total_reward'], summary2['total_reward']),
            ("Longueur épisode", summary1['episode_length'], summary2['episode_length']),
            ("Efficacité", f"{summary1['efficiency']:.3f}", f"{summary2['efficiency']:.3f}"),
            ("Succès", "✅" if summary1['success'] else "❌", "✅" if summary2['success'] else "❌")
        ]
        
        for metric_name, val1, val2 in metrics:
            if metric_name in ["Récompense totale", "Efficacité"]:
                # Plus élevé = meilleur
                better = "A" if float(str(val1).replace("✅", "1").replace("❌", "0")) > float(str(val2).replace("✅", "1").replace("❌", "0")) else "B" if float(str(val1).replace("✅", "1").replace("❌", "0")) < float(str(val2).replace("✅", "1").replace("❌", "0")) else "="
            elif metric_name == "Longueur épisode":
                # Plus court = meilleur (plus efficace)
                better = "A" if val1 < val2 else "B" if val1 > val2 else "="
            else:
                better = "-"
            
            print(f"{metric_name:<20} {str(val1):<15} {str(val2):<15} {better:<10}")
        
        # Actions utilisées
        print(f"\n🎯 ACTIONS UTILISÉES:")
        actions_names = ["Sud", "Nord", "Est", "Ouest", "Prendre", "Déposer"]
        print(f"{'Action':<12} {'Traj A':<8} {'Traj B':<8}")
        print("-" * 30)
        for action in actions_names:
            count1 = summary1['action_counts'].get(action, 0)
            count2 = summary2['action_counts'].get(action, 0)
            print(f"{action:<12} {count1:<8} {count2:<8}")
        
        # Séquence des premières actions
        print(f"\n🔄 PREMIÈRES 10 ACTIONS:")
        seq1 = " → ".join(summary1['actions_taken'][:10])
        seq2 = " → ".join(summary2['actions_taken'][:10])
        print(f"Trajectoire A: {seq1}")
        print(f"Trajectoire B: {seq2}")
        
        print("="*80)
    
    def visualize_trajectories(self, traj1: Trajectory, traj2: Trajectory, 
                             save_path: str = None):
        """
        Crée une visualisation graphique des deux trajectoires
        
        Args:
            traj1: Première trajectoire
            traj2: Deuxième trajectoire
            save_path: Chemin pour sauvegarder le graphique
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Récompenses au fil du temps
        axes[0, 0].plot([step.reward for step in traj1.steps], 
                       label='Trajectoire A', marker='o', alpha=0.7)
        axes[0, 0].plot([step.reward for step in traj2.steps], 
                       label='Trajectoire B', marker='s', alpha=0.7)
        axes[0, 0].set_title('Récompenses par pas de temps')
        axes[0, 0].set_xlabel('Pas de temps')
        axes[0, 0].set_ylabel('Récompense')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Récompenses cumulées
        rewards1_cum = np.cumsum([step.reward for step in traj1.steps])
        rewards2_cum = np.cumsum([step.reward for step in traj2.steps])
        axes[0, 1].plot(rewards1_cum, label='Trajectoire A', linewidth=2)
        axes[0, 1].plot(rewards2_cum, label='Trajectoire B', linewidth=2)
        axes[0, 1].set_title('Récompenses cumulées')
        axes[0, 1].set_xlabel('Pas de temps')
        axes[0, 1].set_ylabel('Récompense cumulée')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution des actions
        actions_names = ["Sud", "Nord", "Est", "Ouest", "Prendre", "Déposer"]
        summary1 = self.get_trajectory_summary(traj1)
        summary2 = self.get_trajectory_summary(traj2)
        
        x = np.arange(len(actions_names))
        width = 0.35
        
        counts1 = [summary1['action_counts'].get(action, 0) for action in actions_names]
        counts2 = [summary2['action_counts'].get(action, 0) for action in actions_names]
        
        axes[1, 0].bar(x - width/2, counts1, width, label='Trajectoire A', alpha=0.8)
        axes[1, 0].bar(x + width/2, counts2, width, label='Trajectoire B', alpha=0.8)
        axes[1, 0].set_title('Distribution des actions')
        axes[1, 0].set_xlabel('Actions')
        axes[1, 0].set_ylabel('Nombre d\'utilisations')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(actions_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Résumé textuel
        axes[1, 1].text(0.1, 0.8, f"TRAJECTOIRE A", fontsize=14, fontweight='bold',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Récompense: {traj1.total_reward}", 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Longueur: {traj1.episode_length} pas", 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Efficacité: {summary1['efficiency']:.3f}", 
                       transform=axes[1, 1].transAxes)
        
        axes[1, 1].text(0.1, 0.3, f"TRAJECTOIRE B", fontsize=14, fontweight='bold',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f"Récompense: {traj2.total_reward}", 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.1, f"Longueur: {traj2.episode_length} pas", 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.0, f"Efficacité: {summary2['efficiency']:.3f}", 
                       transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualisation sauvegardée: {save_path}")
        
        plt.show()
    
    def save_trajectories(self, filepath: str):
        """Sauvegarde toutes les trajectoires collectées"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.trajectories, f)
        print(f"Trajectoires sauvegardées: {filepath}")
    
    def load_trajectories(self, filepath: str):
        """Charge les trajectoires depuis un fichier"""
        with open(filepath, 'rb') as f:
            self.trajectories = pickle.load(f)
        self.trajectory_counter = len(self.trajectories)
        print(f"Trajectoires chargées: {filepath}")
    
    def replay_trajectory_visual(self, env, trajectory: Trajectory, 
                                label: str = "Trajectoire", 
                                delay: float = 0.5):
        """
        Rejoue visuellement une trajectoire dans l'environnement avec rendering
        
        Args:
            env: Environnement Gymnasium (doit supporter render_mode='human')
            trajectory: Trajectoire à rejouer
            label: Label pour identifier la trajectoire
            delay: Délai en secondes entre chaque action
        """
        import time
        
        print(f"\n🎬 Replay visuel: {label}")
        print(f"   Longueur: {trajectory.episode_length} pas")
        print(f"   Récompense totale: {trajectory.total_reward}")
        print("   Appuyez sur Ctrl+C pour passer...")
        
        try:
            # Reset l'environnement
            state, _ = env.reset()
            
            # Rejouer chaque action
            for i, step in enumerate(trajectory.steps):
                # Afficher l'état actuel
                env.render()
                
                print(f"   Pas {i+1}/{trajectory.episode_length}: Action={self._action_to_name(step.action)}, Récompense={step.reward}")
                
                # Attendre pour que l'utilisateur puisse voir
                time.sleep(delay)
                
                # Exécuter l'action
                next_state, reward, terminated, truncated, _ = env.step(step.action)
                
                if terminated or truncated:
                    # Afficher l'état final
                    env.render()
                    time.sleep(delay * 2)  # Pause plus longue à la fin
                    break
            
            print(f"✅ Replay terminé: {label}\n")
            
        except KeyboardInterrupt:
            print(f"\n⏭️  Replay interrompu par l'utilisateur\n")
    
    def _action_to_name(self, action: int) -> str:
        """Convertit un numéro d'action en nom lisible"""
        actions_names = ["Sud", "Nord", "Est", "Ouest", "Prendre", "Déposer"]
        return actions_names[action] if 0 <= action < len(actions_names) else f"Action{action}"