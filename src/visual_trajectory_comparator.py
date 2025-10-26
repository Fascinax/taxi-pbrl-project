"""
Module de visualisation graphique pour comparer deux trajectoires Taxi-v3
Utilise le rendu Gymnasium pour afficher les trajectoires côte à côte
"""

import gymnasium as gym
import numpy as np
import pygame
from typing import List
from src.trajectory_manager import Trajectory, TrajectoryStep
import time


class VisualTrajectoryComparator:
    """
    Visualise deux trajectoires Taxi-v3 côte à côte avec rendu Gymnasium
    """
    
    def __init__(self, window_width: int = 1200, window_height: int = 600):
        """
        Initialise le comparateur visuel
        
        Args:
            window_width: Largeur de la fenêtre totale
            window_height: Hauteur de la fenêtre
        """
        self.window_width = window_width
        self.window_height = window_height
        self.cell_size = 100  # Taille d'une cellule dans la grille Taxi
        
        # Initialisation de pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Comparaison de Trajectoires - Taxi-v3")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
    def replay_trajectories_side_by_side(self, traj1: Trajectory, traj2: Trajectory, 
                                         delay: float = 0.3):
        """
        Rejoue deux trajectoires côte à côte de manière synchronisée
        
        Args:
            traj1: Première trajectoire (Trajectoire A)
            traj2: Deuxième trajectoire (Trajectoire B)
            delay: Délai en secondes entre chaque pas
        """
        # Créer deux environnements pour le rendu
        env1 = gym.make("Taxi-v3", render_mode="rgb_array")
        env2 = gym.make("Taxi-v3", render_mode="rgb_array")
        
        # Initialiser les environnements (requis avant de pouvoir render)
        env1.reset()
        env2.reset()
        
        print("\n🎬 Démarrage de la visualisation des trajectoires...")
        print("Fermez la fenêtre ou appuyez sur ÉCHAP pour passer au choix.")
        
        # Longueur maximale pour synchroniser
        max_steps = max(len(traj1.steps), len(traj2.steps))
        
        running = True
        step_idx = 0
        
        while running and step_idx < max_steps:
            # Gestion des événements pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Pause/Play
                        paused = True
                        while paused:
                            for pause_event in pygame.event.get():
                                if pause_event.type == pygame.QUIT:
                                    running = False
                                    paused = False
                                elif pause_event.type == pygame.KEYDOWN:
                                    if pause_event.key == pygame.K_SPACE:
                                        paused = False
                                    elif pause_event.key == pygame.K_ESCAPE:
                                        running = False
                                        paused = False
            
            if not running:
                break
            
            # Effacer l'écran
            self.screen.fill((20, 20, 30))
            
            # Rejouer les pas pour chaque trajectoire
            if step_idx < len(traj1.steps):
                step1 = traj1.steps[step_idx]
                env1.unwrapped.s = step1.state
                frame1 = env1.render()
            else:
                # Si traj1 est terminée, afficher le dernier état
                frame1 = env1.render()
            
            if step_idx < len(traj2.steps):
                step2 = traj2.steps[step_idx]
                env2.unwrapped.s = step2.state
                frame2 = env2.render()
            else:
                # Si traj2 est terminée, afficher le dernier état
                frame2 = env2.render()
            
            # Afficher les frames côte à côte
            self._render_comparison_view(frame1, frame2, step_idx, traj1, traj2)
            
            # Mettre à jour l'affichage
            pygame.display.flip()
            
            # Délai entre les pas
            time.sleep(delay)
            step_idx += 1
        
        # Afficher l'écran final avec les statistiques complètes
        self._render_final_summary(traj1, traj2)
        
        # Attendre que l'utilisateur ferme ou appuie sur une touche
        waiting = True
        print("\n✅ Visualisation terminée. Fermez la fenêtre ou appuyez sur une touche pour continuer...")
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
        
        env1.close()
        env2.close()
    
    def _render_comparison_view(self, frame1: np.ndarray, frame2: np.ndarray, 
                               step_idx: int, traj1: Trajectory, traj2: Trajectory):
        """
        Affiche les deux frames côte à côte avec les informations
        
        Args:
            frame1: Frame de la trajectoire A
            frame2: Frame de la trajectoire B
            step_idx: Index du pas actuel
            traj1: Trajectoire A
            traj2: Trajectoire B
        """
        # Convertir les frames numpy en surfaces pygame
        surf1 = pygame.surfarray.make_surface(np.transpose(frame1, (1, 0, 2)))
        surf2 = pygame.surfarray.make_surface(np.transpose(frame2, (1, 0, 2)))
        
        # Redimensionner pour tenir dans la fenêtre
        target_width = self.window_width // 2 - 60
        target_height = int(target_width * frame1.shape[0] / frame1.shape[1])
        
        surf1_scaled = pygame.transform.scale(surf1, (target_width, target_height))
        surf2_scaled = pygame.transform.scale(surf2, (target_width, target_height))
        
        # Position des frames
        y_offset = 80
        self.screen.blit(surf1_scaled, (30, y_offset))
        self.screen.blit(surf2_scaled, (self.window_width // 2 + 30, y_offset))
        
        # Titre de chaque trajectoire
        title1 = self.font.render("TRAJECTOIRE A", True, (0, 255, 255))
        title2 = self.font.render("TRAJECTOIRE B", True, (255, 165, 0))
        self.screen.blit(title1, (30, 20))
        self.screen.blit(title2, (self.window_width // 2 + 30, 20))
        
        # Informations en temps réel
        info_y = y_offset + target_height + 20
        
        # Trajectoire A
        if step_idx < len(traj1.steps):
            step1 = traj1.steps[step_idx]
            reward1 = step1.reward
            cumul1 = sum(s.reward for s in traj1.steps[:step_idx+1])
            action1 = self._get_action_name(step1.action)
            
            info1_text = [
                f"Pas: {step_idx + 1}/{len(traj1.steps)}",
                f"Action: {action1}",
                f"Récompense: {reward1:+.1f}",
                f"Cumulée: {cumul1:+.1f}"
            ]
        else:
            info1_text = [
                f"TERMINÉE ({len(traj1.steps)} pas)",
                f"Récompense totale: {traj1.total_reward:+.1f}",
                f"Efficacité: {traj1.total_reward/traj1.episode_length:.3f}"
            ]
        
        for i, text in enumerate(info1_text):
            color = (150, 255, 255) if step_idx < len(traj1.steps) else (100, 200, 100)
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (30, info_y + i * 30))
        
        # Trajectoire B
        if step_idx < len(traj2.steps):
            step2 = traj2.steps[step_idx]
            reward2 = step2.reward
            cumul2 = sum(s.reward for s in traj2.steps[:step_idx+1])
            action2 = self._get_action_name(step2.action)
            
            info2_text = [
                f"Pas: {step_idx + 1}/{len(traj2.steps)}",
                f"Action: {action2}",
                f"Récompense: {reward2:+.1f}",
                f"Cumulée: {cumul2:+.1f}"
            ]
        else:
            info2_text = [
                f"TERMINÉE ({len(traj2.steps)} pas)",
                f"Récompense totale: {traj2.total_reward:+.1f}",
                f"Efficacité: {traj2.total_reward/traj2.episode_length:.3f}"
            ]
        
        for i, text in enumerate(info2_text):
            color = (255, 200, 150) if step_idx < len(traj2.steps) else (100, 200, 100)
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (self.window_width // 2 + 30, info_y + i * 30))
        
        # Instructions en bas
        instructions = "ESPACE: Pause/Play  |  ÉCHAP: Passer au choix"
        instr_rendered = self.font_small.render(instructions, True, (200, 200, 200))
        self.screen.blit(instr_rendered, (self.window_width // 2 - 250, self.window_height - 30))
    
    def _render_final_summary(self, traj1: Trajectory, traj2: Trajectory):
        """
        Affiche le résumé final des deux trajectoires
        
        Args:
            traj1: Trajectoire A
            traj2: Trajectoire B
        """
        self.screen.fill((20, 20, 30))
        
        # Titre
        title = self.font.render("RÉSUMÉ DE LA COMPARAISON", True, (255, 255, 100))
        self.screen.blit(title, (self.window_width // 2 - 200, 30))
        
        # Statistiques comparatives
        stats_y = 120
        stats = [
            ("", "Trajectoire A", "Trajectoire B", "Meilleure"),
            ("─" * 80, "", "", ""),
            ("Récompense totale", f"{traj1.total_reward:+.1f}", f"{traj2.total_reward:+.1f}", 
             "A" if traj1.total_reward > traj2.total_reward else "B" if traj1.total_reward < traj2.total_reward else "="),
            ("Longueur", f"{traj1.episode_length}", f"{traj2.episode_length}",
             "A" if traj1.episode_length < traj2.episode_length else "B" if traj1.episode_length > traj2.episode_length else "="),
            ("Efficacité", f"{traj1.total_reward/traj1.episode_length:.3f}", 
             f"{traj2.total_reward/traj2.episode_length:.3f}",
             "A" if traj1.total_reward/traj1.episode_length > traj2.total_reward/traj2.episode_length else 
             "B" if traj1.total_reward/traj1.episode_length < traj2.total_reward/traj2.episode_length else "="),
            ("Succès", "✓" if traj1.total_reward > 0 else "✗", 
             "✓" if traj2.total_reward > 0 else "✗", "-")
        ]
        
        for i, (metric, val_a, val_b, better) in enumerate(stats):
            if i == 0:
                # En-tête
                color = (255, 255, 150)
                text = self.font_small.render(f"{metric:<25} {val_a:<20} {val_b:<20} {better}", True, color)
            elif i == 1:
                # Ligne de séparation
                color = (100, 100, 100)
                text = self.font_small.render(metric, True, color)
            else:
                # Données
                color_a = (150, 255, 255) if better == "A" else (200, 200, 200)
                color_b = (255, 200, 150) if better == "B" else (200, 200, 200)
                
                metric_text = self.font_small.render(metric, True, (200, 200, 200))
                val_a_text = self.font_small.render(val_a, True, color_a)
                val_b_text = self.font_small.render(val_b, True, color_b)
                better_text = self.font_small.render(better, True, (100, 255, 100) if better != "-" else (200, 200, 200))
                
                self.screen.blit(metric_text, (50, stats_y + i * 40))
                self.screen.blit(val_a_text, (300, stats_y + i * 40))
                self.screen.blit(val_b_text, (550, stats_y + i * 40))
                self.screen.blit(better_text, (800, stats_y + i * 40))
                continue
            
            self.screen.blit(text, (50, stats_y + i * 40))
        
        # Instructions
        instructions = "Fermez la fenêtre ou appuyez sur une touche pour faire votre choix"
        instr_rendered = self.font_small.render(instructions, True, (255, 255, 100))
        self.screen.blit(instr_rendered, (self.window_width // 2 - 330, self.window_height - 50))
        
        pygame.display.flip()
    
    def _get_action_name(self, action: int) -> str:
        """
        Retourne le nom lisible d'une action
        
        Args:
            action: Index de l'action
            
        Returns:
            Nom de l'action
        """
        actions = {
            0: "Sud ↓",
            1: "Nord ↑",
            2: "Est →",
            3: "Ouest ←",
            4: "Prendre 🚖",
            5: "Déposer 🎯"
        }
        return actions.get(action, f"Action {action}")
    
    def close(self):
        """Ferme la fenêtre pygame"""
        pygame.quit()
