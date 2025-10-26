"""
Script pour rejouer visuellement des trajectoires sauvegardées
Utilise le nouveau visualiseur Gymnasium
"""

import pickle
import os
from src.visual_trajectory_comparator import VisualTrajectoryComparator

def test_visual_replay():
    """Test de replay visuel de trajectoires sauvegardées"""
    
    trajectory_file = "results/demo_trajectories.pkl"
    
    if not os.path.exists(trajectory_file):
        print(f"❌ Fichier de trajectoires introuvable: {trajectory_file}")
        print("💡 Astuce: Exécutez d'abord 'python demo_preferences.py' pour générer des trajectoires.")
        return
    
    print(f"📦 Chargement des trajectoires depuis {trajectory_file}...")
    
    with open(trajectory_file, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"✅ {len(trajectories)} trajectoires chargées!")
    
    # Afficher les statistiques
    print("\n📊 Statistiques des trajectoires:")
    for i, traj in enumerate(trajectories):
        efficiency = traj.total_reward / traj.episode_length if traj.episode_length > 0 else 0
        success = "✓" if traj.total_reward > 0 else "✗"
        print(f"  {i+1}. Récompense: {traj.total_reward:+6.1f} | Longueur: {traj.episode_length:3d} | Efficacité: {efficiency:.3f} | Succès: {success}")
    
    # Sélectionner deux trajectoires à comparer
    print("\n🎬 Sélection de trajectoires pour visualisation...")
    
    # Trouver la meilleure et la pire
    sorted_trajs = sorted(trajectories, key=lambda t: t.total_reward, reverse=True)
    best = sorted_trajs[0]
    worst = sorted_trajs[-1]
    
    print(f"\n🏆 Meilleure trajectoire: {best.total_reward:+.1f} points en {best.episode_length} pas")
    print(f"💀 Pire trajectoire: {worst.total_reward:+.1f} points en {worst.episode_length} pas")
    
    # Demander à l'utilisateur
    print("\n❓ Quelle comparaison voulez-vous voir ?")
    print("1️⃣  - Meilleure vs Pire")
    print("2️⃣  - Deux trajectoires aléatoires")
    print("3️⃣  - Choisir manuellement les indices")
    print("0️⃣  - Quitter")
    
    choice = input("\n👉 Votre choix: ").strip()
    
    if choice == "1":
        traj1, traj2 = best, worst
        print(f"\n🎬 Visualisation: Meilleure vs Pire")
    elif choice == "2":
        import random
        traj1, traj2 = random.sample(trajectories, 2)
        print(f"\n🎬 Visualisation: Trajectoires aléatoires")
    elif choice == "3":
        try:
            idx1 = int(input(f"Index de la première trajectoire (1-{len(trajectories)}): ")) - 1
            idx2 = int(input(f"Index de la deuxième trajectoire (1-{len(trajectories)}): ")) - 1
            traj1 = trajectories[idx1]
            traj2 = trajectories[idx2]
            print(f"\n🎬 Visualisation: Trajectoire {idx1+1} vs Trajectoire {idx2+2}")
        except (ValueError, IndexError):
            print("❌ Indices invalides. Abandon.")
            return
    else:
        print("👋 Au revoir!")
        return
    
    # Lancer la visualisation
    print("\n🚀 Lancement de la visualisation...")
    print("💡 Contrôles: ESPACE (pause), ÉCHAP (fermer)")
    
    visualizer = VisualTrajectoryComparator()
    visualizer.replay_trajectories_side_by_side(traj1, traj2, delay=0.3)
    visualizer.close()
    
    print("\n✅ Visualisation terminée!")
    
    # Proposer de recommencer
    again = input("\n🔄 Voir une autre comparaison ? (y/n): ").strip().lower()
    if again in ['y', 'yes', 'o', 'oui']:
        test_visual_replay()

if __name__ == "__main__":
    try:
        test_visual_replay()
    except KeyboardInterrupt:
        print("\n\n👋 Interruption utilisateur. Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
