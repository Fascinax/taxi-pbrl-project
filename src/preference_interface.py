from typing import Tuple, List, Dict, Any
import json
import os
from datetime import datetime
from src.trajectory_manager import Trajectory, TrajectoryManager
from src.visual_trajectory_comparator import VisualTrajectoryComparator

class PreferenceInterface:
    """
    Interface pour collecter les préférences entre trajectoires
    """
    
    def __init__(self):
        self.preferences: List[Dict[str, Any]] = []
        self.preference_counter = 0
    
    def collect_preference_interactive(self, traj1: Trajectory, traj2: Trajectory, 
                                     trajectory_manager: TrajectoryManager, 
                                     use_visual: bool = True) -> int:
        """
        Interface interactive pour collecter une préférence entre deux trajectoires
        
        Args:
            traj1: Première trajectoire
            traj2: Deuxième trajectoire
            trajectory_manager: Manager pour affichage des trajectoires
            use_visual: Si True, affiche la visualisation Gymnasium (défaut: True)
            
        Returns:
            int: Choix de l'utilisateur (1 pour traj1, 2 pour traj2, 0 pour égalité)
        """
        print("\n" + "="*100)
        print("🤖 SYSTÈME DE PRÉFÉRENCES - ÉVALUATION DE TRAJECTOIRES")
        print("="*100)
        
        # Visualisation graphique Gymnasium (par défaut)
        if use_visual:
            try:
                visualizer = VisualTrajectoryComparator()
                visualizer.replay_trajectories_side_by_side(traj1, traj2, delay=0.3)
                visualizer.close()
            except Exception as e:
                print(f"⚠️ Erreur lors de la visualisation graphique: {e}")
                print("📊 Passage à l'affichage textuel...")
                use_visual = False
        
        # Affichage de la comparaison textuelle (toujours disponible)
        if not use_visual:
            trajectory_manager.display_trajectory_comparison(traj1, traj2)
        
        # Demande de préférence
        print("\n" + "🔥 VOTRE CHOIX:")
        print("Quelle trajectoire préférez-vous ?")
        print("💡 Critères à considérer: efficacité, récompense, style de navigation, succès...")
        print("")
        print("1️⃣  - Je préfère la TRAJECTOIRE A (cyan)")
        print("2️⃣  - Je préfère la TRAJECTOIRE B (orange)") 
        print("0️⃣  - Les deux sont équivalentes (égalité)")
        print("🆘 - Tapez 'help' pour plus d'informations")
        print("🎯 - Tapez 'replay' pour revoir la visualisation graphique")
        print("📊 - Tapez 'text' pour voir la comparaison textuelle détaillée")
        print("")
        
        while True:
            try:
                choice = input("👉 Votre choix (1/2/0/help/replay/text): ").strip().lower()
                
                if choice == 'help':
                    self._display_help()
                    continue
                elif choice == 'replay':
                    # Rejouer la visualisation Gymnasium
                    try:
                        visualizer = VisualTrajectoryComparator()
                        visualizer.replay_trajectories_side_by_side(traj1, traj2, delay=0.3)
                        visualizer.close()
                    except Exception as e:
                        print(f"⚠️ Erreur lors de la visualisation: {e}")
                    continue
                elif choice == 'text':
                    # Afficher la comparaison textuelle détaillée
                    trajectory_manager.display_trajectory_comparison(traj1, traj2)
                    continue
                elif choice == 'viz':
                    # Rétrocompatibilité avec l'ancien 'viz'
                    trajectory_manager.visualize_trajectories(traj1, traj2)
                    continue
                elif choice in ['1', '2', '0']:
                    choice_int = int(choice)
                    
                    # Demande de justification (optionnelle)
                    reasoning = input("💭 Pourquoi ce choix ? (optionnel): ").strip()
                    if not reasoning:
                        reasoning = "Aucune justification fournie"
                    
                    # Enregistrement de la préférence
                    preference = self._record_preference(traj1, traj2, choice_int, reasoning)
                    
                    # Confirmation
                    choice_names = {1: "Trajectoire A", 2: "Trajectoire B", 0: "Égalité"}
                    print(f"\n✅ Préférence enregistrée: {choice_names[choice_int]}")
                    if reasoning != "Aucune justification fournie":
                        print(f"📝 Justification: {reasoning}")
                    
                    return choice_int
                else:
                    print("❌ Choix invalide. Utilisez 1, 2, 0, 'help', 'replay' ou 'text'")
                    
            except (ValueError, KeyboardInterrupt):
                print("❌ Entrée invalide ou interruption. Réessayez.")
    
    def collect_preference_batch(self, trajectory_pairs: List[Tuple[Trajectory, Trajectory]], 
                               trajectory_manager: TrajectoryManager) -> List[int]:
        """
        Collecte des préférences pour plusieurs paires de trajectoires
        
        Args:
            trajectory_pairs: Liste de paires de trajectoires
            trajectory_manager: Manager pour affichage
            
        Returns:
            List[int]: Liste des préférences
        """
        preferences = []
        total_pairs = len(trajectory_pairs)
        
        print(f"\n🎯 SESSION DE PRÉFÉRENCES: {total_pairs} comparaisons à effectuer")
        
        for i, (traj1, traj2) in enumerate(trajectory_pairs):
            print(f"\n📊 Comparaison {i+1}/{total_pairs}")
            preference = self.collect_preference_interactive(traj1, traj2, trajectory_manager)
            preferences.append(preference)
            
            # Demande de continuation pour les sessions longues
            if i < total_pairs - 1 and (i + 1) % 5 == 0:
                continue_choice = input(f"\n⏸️  Pause après {i+1} comparaisons. Continuer ? (y/n): ").strip().lower()
                if continue_choice in ['n', 'non', 'no']:
                    print(f"🛑 Session interrompue. {i+1} préférences collectées.")
                    break
        
        self._save_session_summary(preferences, total_pairs)
        return preferences
    
    def _display_help(self):
        """Affiche l'aide pour le système de préférences"""
        print("\n" + "🆘 AIDE - SYSTÈME DE PRÉFÉRENCES")
        print("-" * 50)
        print("🎯 OBJECTIF: Vous aidez l'agent à apprendre vos préférences")
        print("   en comparant différentes façons de résoudre la tâche Taxi.")
        print("")
        print("📊 CRITÈRES DE COMPARAISON:")
        print("   • Récompense totale: Plus élevée = mieux")
        print("   • Efficacité: Récompense/pas de temps")
        print("   • Longueur épisode: Plus court peut être mieux (mais pas toujours)")
        print("   • Style de navigation: Certains préfèrent des trajets directs")
        print("   • Succès: L'agent a-t-il réussi la tâche ?")
        print("")
        print("💡 CONSEILS:")
        print("   • Suivez votre intuition sur ce qui vous semble 'mieux'")
        print("   • Vous pouvez valoriser la sécurité, la vitesse, l'élégance...")
        print("   • Les égalités (0) sont acceptables si vraiment équivalent")
        print("")
        print("🎮 ACTIONS TAXI:")
        print("   • Sud/Nord/Est/Ouest: Déplacement")
        print("   • Prendre: Récupérer le passager")
        print("   • Déposer: Déposer le passager à destination")
        print("-" * 50)
    
    def _record_preference(self, traj1: Trajectory, traj2: Trajectory, 
                          choice: int, reasoning: str) -> Dict[str, Any]:
        """
        Enregistre une préférence dans la base de données
        
        Args:
            traj1: Première trajectoire
            traj2: Deuxième trajectoire  
            choice: Choix utilisateur (1, 2, ou 0)
            reasoning: Justification du choix
            
        Returns:
            Dict contenant la préférence enregistrée
        """
        preference = {
            'preference_id': self.preference_counter,
            'timestamp': datetime.now().isoformat(),
            'trajectory_a_id': traj1.episode_id,
            'trajectory_b_id': traj2.episode_id,
            'trajectory_a_reward': traj1.total_reward,
            'trajectory_b_reward': traj2.total_reward,
            'trajectory_a_length': traj1.episode_length,
            'trajectory_b_length': traj2.episode_length,
            'choice': choice,  # 1=traj1, 2=traj2, 0=equal
            'reasoning': reasoning,
            'trajectory_a_efficiency': traj1.total_reward / traj1.episode_length if traj1.episode_length > 0 else 0,
            'trajectory_b_efficiency': traj2.total_reward / traj2.episode_length if traj2.episode_length > 0 else 0
        }
        
        self.preferences.append(preference)
        self.preference_counter += 1
        
        return preference
    
    def _save_session_summary(self, preferences: List[int], total_pairs: int):
        """Sauvegarde un résumé de la session"""
        completed = len(preferences)
        choice_counts = {0: 0, 1: 0, 2: 0}
        for choice in preferences:
            choice_counts[choice] += 1
        
        print(f"\n📋 RÉSUMÉ DE SESSION:")
        print(f"   • Comparaisons complétées: {completed}/{total_pairs}")
        print(f"   • Préférences Trajectoire A: {choice_counts[1]}")
        print(f"   • Préférences Trajectoire B: {choice_counts[2]}")
        print(f"   • Égalités: {choice_counts[0]}")
        
    def get_preference_statistics(self) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les préférences collectées
        
        Returns:
            Dict contenant les statistiques
        """
        if not self.preferences:
            return {"error": "Aucune préférence collectée"}
        
        total_prefs = len(self.preferences)
        choice_counts = {0: 0, 1: 0, 2: 0}
        
        for pref in self.preferences:
            choice_counts[pref['choice']] += 1
        
        # Analyse des critères de préférence basés sur les données
        reward_preferences = []  # Préfère-t-on les récompenses élevées ?
        efficiency_preferences = []  # Préfère-t-on l'efficacité ?
        
        for pref in self.preferences:
            if pref['choice'] in [1, 2]:  # Pas d'égalité
                preferred_traj = 'a' if pref['choice'] == 1 else 'b'
                
                # Comparaison récompenses
                reward_a = pref['trajectory_a_reward']
                reward_b = pref['trajectory_b_reward']
                if reward_a != reward_b:
                    prefers_higher_reward = (preferred_traj == 'a' and reward_a > reward_b) or \
                                          (preferred_traj == 'b' and reward_b > reward_a)
                    reward_preferences.append(prefers_higher_reward)
                
                # Comparaison efficacité
                eff_a = pref['trajectory_a_efficiency']
                eff_b = pref['trajectory_b_efficiency']
                if abs(eff_a - eff_b) > 0.01:  # Différence significative
                    prefers_higher_efficiency = (preferred_traj == 'a' and eff_a > eff_b) or \
                                              (preferred_traj == 'b' and eff_b > eff_a)
                    efficiency_preferences.append(prefers_higher_efficiency)
        
        stats = {
            'total_preferences': total_prefs,
            'choice_distribution': {
                'trajectory_a': choice_counts[1],
                'trajectory_b': choice_counts[2], 
                'equal': choice_counts[0]
            },
            'choice_percentages': {
                'trajectory_a': (choice_counts[1] / total_prefs * 100) if total_prefs > 0 else 0,
                'trajectory_b': (choice_counts[2] / total_prefs * 100) if total_prefs > 0 else 0,
                'equal': (choice_counts[0] / total_prefs * 100) if total_prefs > 0 else 0
            },
            'reward_preference_tendency': sum(reward_preferences) / len(reward_preferences) * 100 if reward_preferences else 0,
            'efficiency_preference_tendency': sum(efficiency_preferences) / len(efficiency_preferences) * 100 if efficiency_preferences else 0
        }
        
        return stats
    
    def save_preferences(self, filepath: str):
        """Sauvegarde les préférences dans un fichier JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'preferences': self.preferences,
            'statistics': self.get_preference_statistics(),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_preferences': len(self.preferences)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Préférences sauvegardées: {filepath}")
    
    def load_preferences(self, filepath: str):
        """Charge les préférences depuis un fichier JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.preferences = data['preferences']
        self.preference_counter = len(self.preferences)
        
        print(f"Préférences chargées: {filepath}")
        
    def display_preferences_summary(self):
        """Affiche un résumé des préférences collectées"""
        stats = self.get_preference_statistics()
        
        if 'error' in stats:
            print(stats['error'])
            return
        
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DES PRÉFÉRENCES COLLECTÉES")
        print("="*60)
        print(f"Total de préférences: {stats['total_preferences']}")
        print(f"Trajectoire A préférée: {stats['choice_distribution']['trajectory_a']} fois ({stats['choice_percentages']['trajectory_a']:.1f}%)")
        print(f"Trajectoire B préférée: {stats['choice_distribution']['trajectory_b']} fois ({stats['choice_percentages']['trajectory_b']:.1f}%)")
        print(f"Égalités déclarées: {stats['choice_distribution']['equal']} fois ({stats['choice_percentages']['equal']:.1f}%)")
        
        print(f"\n🎯 TENDANCES DÉTECTÉES:")
        print(f"Préférence pour récompenses élevées: {stats['reward_preference_tendency']:.1f}%")
        print(f"Préférence pour efficacité élevée: {stats['efficiency_preference_tendency']:.1f}%")
        print("="*60)