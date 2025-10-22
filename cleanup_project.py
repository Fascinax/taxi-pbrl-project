"""
Script de Nettoyage du Projet PBRL
===================================

Supprime les fichiers temporaires et inutiles tout en préservant
les résultats importants et la documentation.
"""

import os
import shutil
from pathlib import Path

def list_files_to_remove():
    """Liste les fichiers à supprimer"""
    files_to_remove = []
    
    # Fichiers temporaires
    temp_files = [
        "results/mountaincar_preferences_temp.json",
        # Pycache
        "__pycache__",
        "src/__pycache__",
        # Anciens fichiers de log
        "results/workflow_logs",
    ]
    
    # Fichiers de développement obsolètes
    dev_files = [
        "analyze_robust_results.py",
        "collect_robust_preferences.py",
        "debug_trajectories.py",
        "integrate_web_preferences.py",
        "launch_web_interface.py",
        "run_robust_workflow.py",
        "test_comparison_quality.py",
        "test_web_interface.py",
        "train_robust_pbrl.py",
        "app.py",
        "web_static",
        "static",
        "templates",
    ]
    
    # Documentation obsolète
    obsolete_docs = [
        "CHANGELOG_WEB.md",
        "DOC_INDEX.md",
        "PROJECT_STRUCTURE.md",
        "ROBUST_UPGRADE_SUMMARY.md",
        "TROUBLESHOOTING.md",
        "WEB_INTERFACE.md",
        "WEB_SUMMARY.md",
        "MOUNTAINCAR_GUIDE.md",
        "MOUNTAINCAR_PBRL_COMPLETE.md",
        "MOUNTAINCAR_SETUP_COMPLETE.md",
        "docs/correspondance_papier_recherche.md",
        "docs/detailed_analysis.md",
        "docs/final_insights.md",
        "docs/generation_trajectoires_pbrl.md",
        "docs/interface_preview.md",
        "docs/power_analysis.py",
        "docs/web_interface_guide.md",
    ]
    
    files_to_remove.extend(temp_files)
    files_to_remove.extend(dev_files)
    files_to_remove.extend(obsolete_docs)
    
    return files_to_remove

def list_files_to_keep():
    """Liste les fichiers importants à GARDER"""
    keep_files = {
        # Scripts principaux
        "train_classical_agent.py",
        "train_pbrl_agent.py",
        "train_mountaincar_classical.py",
        "train_mountaincar_pbrl.py",
        "demo_preferences.py",
        "demo_mountaincar.py",
        "collect_mountaincar_preferences.py",
        "collect_mountaincar_preferences_auto.py",
        "statistical_analysis.py",
        "compare_taxi_vs_mountaincar.py",
        
        # Code source
        "src/q_learning_agent.py",
        "src/pbrl_agent.py",
        "src/trajectory_manager.py",
        "src/preference_interface.py",
        "src/mountain_car_discretizer.py",
        "src/mountain_car_agent.py",
        "src/mountain_car_pbrl_agent.py",
        
        # Documentation
        "README.md",
        "GUIDE_UTILISATION.md",
        "QUICKSTART.md",
        "MOUNTAINCAR_RESULTS_FINAL.md",
        "docs/rapport_final.md",
        "docs/customizing_pairs.md",
        "requirements.txt",
        
        # Résultats Taxi
        "results/q_learning_agent_classical.pkl",
        "results/pbrl_agent.pkl",
        "results/comparison_classical_vs_pbrl.png",
        "results/detailed_comparison.json",
        "results/advanced_statistical_analysis.png",
        "results/performance_report.md",
        "results/demo_trajectories.pkl",
        
        # Résultats MountainCar
        "results/mountain_car_agent_classical.pkl",
        "results/mountain_car_agent_pbrl.pkl",
        "results/comparison_mountaincar_classical_vs_pbrl.png",
        "results/mountaincar_pbrl_comparison.json",
        "results/mountaincar_preferences.json",
        "results/mountaincar_trajectories.pkl",
        "results/training_progress_mountaincar.png",
        
        # Comparaison
        "results/comparison_taxi_vs_mountaincar_pbrl.png",
        "results/comparison_insights.txt",
        "results/comparison_taxi_vs_mountaincar.json",
    }
    
    return keep_files

def clean_project(dry_run=False):
    """Nettoie le projet en supprimant les fichiers inutiles"""
    
    project_root = Path(".")
    files_to_remove = list_files_to_remove()
    files_to_keep = list_files_to_keep()
    
    removed_count = 0
    kept_count = 0
    not_found = []
    
    print("=" * 80)
    print("🧹 NETTOYAGE DU PROJET PBRL")
    print("=" * 80)
    print()
    
    if dry_run:
        print("⚠️  MODE DRY-RUN : Aucun fichier ne sera supprimé")
        print("   Exécutez avec dry_run=False pour supprimer réellement")
        print()
    
    print("📋 Analyse des fichiers...")
    print()
    
    # Traiter chaque fichier
    for file_path_str in files_to_remove:
        file_path = project_root / file_path_str
        
        if file_path.exists():
            # Vérifier si dans la liste des fichiers à garder
            if file_path_str in files_to_keep:
                print(f"✅ GARDER : {file_path_str} (dans la liste de conservation)")
                kept_count += 1
                continue
            
            # Supprimer
            if dry_run:
                print(f"🗑️  SUPPRIMER : {file_path_str}")
            else:
                try:
                    if file_path.is_dir():
                        shutil.rmtree(file_path)
                    else:
                        file_path.unlink()
                    print(f"✅ SUPPRIMÉ : {file_path_str}")
                except Exception as e:
                    print(f"❌ ERREUR : {file_path_str} - {e}")
                    continue
            
            removed_count += 1
        else:
            not_found.append(file_path_str)
    
    print()
    print("=" * 80)
    print("📊 RÉSUMÉ DU NETTOYAGE")
    print("=" * 80)
    print()
    
    if dry_run:
        print(f"🗑️  Fichiers à supprimer : {removed_count}")
    else:
        print(f"✅ Fichiers supprimés : {removed_count}")
    
    print(f"✅ Fichiers conservés : {kept_count}")
    print(f"⚠️  Fichiers non trouvés : {len(not_found)}")
    print()
    
    if not_found and len(not_found) <= 10:
        print("📝 Fichiers non trouvés (déjà supprimés ou jamais créés) :")
        for nf in not_found:
            print(f"   - {nf}")
        print()
    
    # Statistiques de l'espace
    print("=" * 80)
    print("📁 STRUCTURE FINALE DU PROJET")
    print("=" * 80)
    print()
    
    print("✅ SCRIPTS PRINCIPAUX (10)")
    scripts = [
        "train_classical_agent.py",
        "train_pbrl_agent.py", 
        "train_mountaincar_classical.py",
        "train_mountaincar_pbrl.py",
        "demo_preferences.py",
        "demo_mountaincar.py",
        "collect_mountaincar_preferences_auto.py",
        "statistical_analysis.py",
        "compare_taxi_vs_mountaincar.py",
        "cleanup_project.py"
    ]
    for script in scripts:
        status = "✅" if (project_root / script).exists() else "❌"
        print(f"   {status} {script}")
    
    print()
    print("✅ CODE SOURCE (7 fichiers)")
    src_files = [
        "src/q_learning_agent.py",
        "src/pbrl_agent.py",
        "src/trajectory_manager.py",
        "src/preference_interface.py",
        "src/mountain_car_discretizer.py",
        "src/mountain_car_agent.py",
        "src/mountain_car_pbrl_agent.py",
    ]
    for src in src_files:
        status = "✅" if (project_root / src).exists() else "❌"
        print(f"   {status} {src}")
    
    print()
    print("✅ DOCUMENTATION (5 fichiers)")
    docs = [
        "README.md",
        "GUIDE_UTILISATION.md",
        "QUICKSTART.md",
        "MOUNTAINCAR_RESULTS_FINAL.md",
        "docs/rapport_final.md",
    ]
    for doc in docs:
        status = "✅" if (project_root / doc).exists() else "❌"
        print(f"   {status} {doc}")
    
    print()
    print("✅ RÉSULTATS IMPORTANTS")
    important_results = [
        "results/comparison_taxi_vs_mountaincar_pbrl.png",
        "results/comparison_insights.txt",
        "results/detailed_comparison.json",
        "results/mountaincar_pbrl_comparison.json",
    ]
    for result in important_results:
        status = "✅" if (project_root / result).exists() else "❌"
        print(f"   {status} {result}")
    
    print()
    print("=" * 80)
    
    if dry_run:
        print()
        print("⚠️  AUCUN FICHIER N'A ÉTÉ SUPPRIMÉ (mode dry-run)")
        print()
        print("Pour supprimer réellement les fichiers, modifiez le script :")
        print("   clean_project(dry_run=False)")
        print()
    else:
        print()
        print("✅ NETTOYAGE TERMINÉ !")
        print()
    
    print("=" * 80)

def main():
    """Point d'entrée principal"""
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                  SCRIPT DE NETTOYAGE DU PROJET PBRL                       ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Mode dry-run par défaut
    dry_run = True
    
    # Demander confirmation si on veut supprimer
    response = input("🔍 Voulez-vous voir ce qui sera supprimé (dry-run) ? [O/n] : ")
    if response.lower() in ['n', 'non', 'no']:
        response = input("⚠️  Voulez-vous VRAIMENT supprimer les fichiers ? [o/N] : ")
        if response.lower() in ['o', 'oui', 'yes', 'y']:
            dry_run = False
        else:
            print("❌ Opération annulée.")
            return
    
    print()
    clean_project(dry_run=dry_run)
    
    if dry_run:
        print()
        print("💡 Pour nettoyer réellement, réexécutez et choisissez 'non' puis 'oui'")
        print()

if __name__ == "__main__":
    main()
