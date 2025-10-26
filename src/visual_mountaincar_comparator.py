"""
Module de visualisation graphique pour comparer deux trajectoires MountainCar-v0
Utilise le rendu Gymnasium pour afficher les trajectoires côte à côte
"""

import gymnasium as gym
import numpy as np
import pygame
from typing import List
import time


class MountainCarTrajectory:
    """Représente une trajectoire MountainCar avec ses métriques"""
    def __init__(self, steps, total_reward, episode_length, episode_id):
        self.steps = steps
        self.total_reward = total_reward
        self.episode_length = episode_length
        self.episode_id = episode_id
        self.max_position = -1.2
        self.max_velocity = 0.0
        self.success = False


class VisualMountainCarComparator:
    """
    Visualise deux trajectoires MountainCar-v0 côte à côte avec rendu Gymnasium
    """
    
    def __init__(self, window_width: int = 1400, window_height: int = 700):
        """
        Initialise le comparateur visuel
        
        Args:
            window_width: Largeur de la fenêtre totale
            window_height: Hauteur de la fenêtre
        """
        self.window_width = window_width
        self.window_height = window_height
        
        # Initialisation de pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Comparaison de Trajectoires - MountainCar-v0")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 20)
        
    def replay_trajectories_side_by_side(self, traj1: MountainCarTrajectory, 
                                         traj2: MountainCarTrajectory, 
                                         delay: float = 0.05):
        """
        Rejoue deux trajectoires côte à côte de manière synchronisée
        
        Args:
            traj1: Première trajectoire (Trajectoire A)
            traj2: Deuxième trajectoire (Trajectoire B)
            delay: Délai en secondes entre chaque pas
        """
        # Créer deux environnements pour le rendu
        env1 = gym.make("MountainCar-v0", render_mode="rgb_array")
        env2 = gym.make("MountainCar-v0", render_mode="rgb_array")
        
        # Initialiser les environnements
        env1.reset()
        env2.reset()
        
        print("\n🎬 Démarrage de la visualisation des trajectoires MountainCar...")
        print("Fermez la fenêtre ou appuyez sur ÉCHAP pour passer au choix.")
        print("ESPACE: Pause/Play | R: Recommencer | 1-5: Vitesse")
        
        # Longueur maximale pour synchroniser
        max_steps = max(len(traj1.steps), len(traj2.steps))
        
        running = True
        step_idx = 0
        paused = False
        speed_multiplier = 1.0
        
        while running and step_idx < max_steps:
            # Gestion des événements pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        step_idx = 0
                    elif event.key == pygame.K_1:
                        speed_multiplier = 0.5
                        print("⚡ Vitesse: 0.5x")
                    elif event.key == pygame.K_2:
                        speed_multiplier = 1.0
                        print("⚡ Vitesse: 1.0x (normale)")
                    elif event.key == pygame.K_3:
                        speed_multiplier = 2.0
                        print("⚡ Vitesse: 2.0x")
                    elif event.key == pygame.K_4:
                        speed_multiplier = 5.0
                        print("⚡ Vitesse: 5.0x")
                    elif event.key == pygame.K_5:
                        speed_multiplier = 10.0
                        print("⚡ Vitesse: 10.0x")
            
            if paused:
                # Afficher "PAUSE"
                self.screen.fill((20, 20, 30))
                pause_text = self.font.render("⏸️ PAUSE - Appuyez sur ESPACE", True, (255, 255, 100))
                self.screen.blit(pause_text, (self.window_width // 2 - 200, self.window_height // 2))
                pygame.display.flip()
                continue
            
            if not running:
                break
            
            # Effacer l'écran
            self.screen.fill((20, 20, 30))
            
            # Rejouer les pas pour chaque trajectoire
            if step_idx < len(traj1.steps):
                step1 = traj1.steps[step_idx]
                # Restaurer l'état continu
                position1, velocity1 = step1.continuous_state
                env1.unwrapped.state = np.array([position1, velocity1])
                frame1 = env1.render()
            else:
                # Si traj1 est terminée, afficher le dernier état
                frame1 = env1.render()
            
            if step_idx < len(traj2.steps):
                step2 = traj2.steps[step_idx]
                # Restaurer l'état continu
                position2, velocity2 = step2.continuous_state
                env2.unwrapped.state = np.array([position2, velocity2])
                frame2 = env2.render()
            else:
                # Si traj2 est terminée, afficher le dernier état
                frame2 = env2.render()
            
            # Afficher les frames côte à côte
            self._render_comparison_view(frame1, frame2, step_idx, traj1, traj2, speed_multiplier)
            
            # Mettre à jour l'affichage
            pygame.display.flip()
            
            # Délai entre les pas (ajusté par la vitesse)
            time.sleep(delay / speed_multiplier)
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
                               step_idx: int, traj1: MountainCarTrajectory, 
                               traj2: MountainCarTrajectory, speed: float):
        """
        Affiche les deux frames côte à côte avec les informations
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
        y_offset = 100
        self.screen.blit(surf1_scaled, (30, y_offset))
        self.screen.blit(surf2_scaled, (self.window_width // 2 + 30, y_offset))
        
        # Titre de chaque trajectoire
        title1 = self.font.render("TRAJECTOIRE A", True, (0, 255, 255))
        title2 = self.font.render("TRAJECTOIRE B", True, (255, 165, 0))
        self.screen.blit(title1, (30, 20))
        self.screen.blit(title2, (self.window_width // 2 + 30, 20))
        
        # Indicateur de succès
        success1_text = "✅ SUCCÈS" if traj1.success else "❌ ÉCHEC"
        success2_text = "✅ SUCCÈS" if traj2.success else "❌ ÉCHEC"
        success1_color = (100, 255, 100) if traj1.success else (255, 100, 100)
        success2_color = (100, 255, 100) if traj2.success else (255, 100, 100)
        
        success1_render = self.font_small.render(success1_text, True, success1_color)
        success2_render = self.font_small.render(success2_text, True, success2_color)
        self.screen.blit(success1_render, (30, 60))
        self.screen.blit(success2_render, (self.window_width // 2 + 30, 60))
        
        # Informations en temps réel
        info_y = y_offset + target_height + 20
        
        # Trajectoire A
        if step_idx < len(traj1.steps):
            step1 = traj1.steps[step_idx]
            position1, velocity1 = step1.continuous_state
            reward1 = step1.reward
            cumul1 = sum(s.reward for s in traj1.steps[:step_idx+1])
            action1 = self._get_action_name(step1.action)
            
            info1_text = [
                f"Pas: {step_idx + 1}/{len(traj1.steps)}",
                f"Action: {action1}",
                f"Position: {position1:.3f}",
                f"Vitesse: {velocity1:+.3f}",
                f"Récompense: {reward1:+.1f}",
                f"Cumulée: {cumul1:+.1f}",
                f"Max Pos: {traj1.max_position:.3f}"
            ]
        else:
            info1_text = [
                f"TERMINÉE ({len(traj1.steps)} pas)",
                f"Récompense: {traj1.total_reward:+.1f}",
                f"Max Pos: {traj1.max_position:.3f}",
                f"Succès: {'Oui 🎯' if traj1.success else 'Non ❌'}"
            ]
        
        for i, text in enumerate(info1_text):
            color = (150, 255, 255) if step_idx < len(traj1.steps) else (100, 200, 100)
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (30, info_y + i * 28))
        
        # Trajectoire B
        if step_idx < len(traj2.steps):
            step2 = traj2.steps[step_idx]
            position2, velocity2 = step2.continuous_state
            reward2 = step2.reward
            cumul2 = sum(s.reward for s in traj2.steps[:step_idx+1])
            action2 = self._get_action_name(step2.action)
            
            info2_text = [
                f"Pas: {step_idx + 1}/{len(traj2.steps)}",
                f"Action: {action2}",
                f"Position: {position2:.3f}",
                f"Vitesse: {velocity2:+.3f}",
                f"Récompense: {reward2:+.1f}",
                f"Cumulée: {cumul2:+.1f}",
                f"Max Pos: {traj2.max_position:.3f}"
            ]
        else:
            info2_text = [
                f"TERMINÉE ({len(traj2.steps)} pas)",
                f"Récompense: {traj2.total_reward:+.1f}",
                f"Max Pos: {traj2.max_position:.3f}",
                f"Succès: {'Oui 🎯' if traj2.success else 'Non ❌'}"
            ]
        
        for i, text in enumerate(info2_text):
            color = (255, 200, 150) if step_idx < len(traj2.steps) else (100, 200, 100)
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (self.window_width // 2 + 30, info_y + i * 28))
        
        # Instructions en bas
        instructions = f"ESPACE: Pause | R: Recommencer | 1-5: Vitesse (x{speed:.1f}) | ÉCHAP: Passer au choix"
        instr_rendered = self.font_tiny.render(instructions, True, (200, 200, 200))
        self.screen.blit(instr_rendered, (self.window_width // 2 - 350, self.window_height - 30))
    
    def _render_final_summary(self, traj1: MountainCarTrajectory, traj2: MountainCarTrajectory):
        """
        Affiche le résumé final des deux trajectoires
        """
        self.screen.fill((20, 20, 30))
        
        # Titre
        title = self.font.render("RÉSUMÉ DE LA COMPARAISON", True, (255, 255, 100))
        self.screen.blit(title, (self.window_width // 2 - 220, 30))
        
        # Statistiques comparatives
        stats_y = 120
        stats = [
            ("", "Trajectoire A", "Trajectoire B", "Meilleure"),
            ("─" * 90, "", "", ""),
            ("Récompense totale", f"{traj1.total_reward:+.1f}", f"{traj2.total_reward:+.1f}", 
             "A" if traj1.total_reward > traj2.total_reward else "B" if traj1.total_reward < traj2.total_reward else "="),
            ("Longueur (pas)", f"{traj1.episode_length}", f"{traj2.episode_length}",
             "A" if traj1.episode_length < traj2.episode_length else "B" if traj1.episode_length > traj2.episode_length else "="),
            ("Position maximale", f"{traj1.max_position:.3f}", f"{traj2.max_position:.3f}",
             "A" if traj1.max_position > traj2.max_position else "B" if traj1.max_position < traj2.max_position else "="),
            ("Vitesse maximale", f"{traj1.max_velocity:.3f}", f"{traj2.max_velocity:.3f}",
             "A" if traj1.max_velocity > traj2.max_velocity else "B" if traj1.max_velocity < traj2.max_velocity else "="),
            ("Succès", "✓" if traj1.success else "✗", 
             "✓" if traj2.success else "✗", 
             "A" if traj1.success and not traj2.success else "B" if traj2.success and not traj1.success else "="),
            ("Efficacité", f"{traj1.total_reward/traj1.episode_length:.3f}", 
             f"{traj2.total_reward/traj2.episode_length:.3f}",
             "A" if traj1.total_reward/traj1.episode_length > traj2.total_reward/traj2.episode_length else 
             "B" if traj1.total_reward/traj1.episode_length < traj2.total_reward/traj2.episode_length else "=")
        ]
        
        for i, (metric, val_a, val_b, better) in enumerate(stats):
            if i == 0:
                # En-tête
                color = (255, 255, 150)
                metric_text = self.font_small.render(metric, True, color)
                val_a_text = self.font_small.render(val_a, True, color)
                val_b_text = self.font_small.render(val_b, True, color)
                better_text = self.font_small.render(better, True, color)
                
                self.screen.blit(metric_text, (50, stats_y + i * 40))
                self.screen.blit(val_a_text, (400, stats_y + i * 40))
                self.screen.blit(val_b_text, (650, stats_y + i * 40))
                self.screen.blit(better_text, (900, stats_y + i * 40))
            elif i == 1:
                # Ligne de séparation
                color = (100, 100, 100)
                text = self.font_small.render(metric, True, color)
                self.screen.blit(text, (50, stats_y + i * 40))
            else:
                # Données
                color_a = (150, 255, 255) if better == "A" else (200, 200, 200)
                color_b = (255, 200, 150) if better == "B" else (200, 200, 200)
                
                metric_text = self.font_small.render(metric, True, (200, 200, 200))
                val_a_text = self.font_small.render(val_a, True, color_a)
                val_b_text = self.font_small.render(val_b, True, color_b)
                better_text = self.font_small.render(better, True, (100, 255, 100) if better != "=" else (200, 200, 200))
                
                self.screen.blit(metric_text, (50, stats_y + i * 40))
                self.screen.blit(val_a_text, (400, stats_y + i * 40))
                self.screen.blit(val_b_text, (650, stats_y + i * 40))
                self.screen.blit(better_text, (900, stats_y + i * 40))
        
        # Instructions
        instructions = "Fermez la fenêtre ou appuyez sur une touche pour faire votre choix"
        instr_rendered = self.font_small.render(instructions, True, (255, 255, 100))
        self.screen.blit(instr_rendered, (self.window_width // 2 - 380, self.window_height - 50))
        
        pygame.display.flip()
    
    def _get_action_name(self, action: int) -> str:
        """
        Retourne le nom lisible d'une action MountainCar
        
        Args:
            action: Index de l'action
            
        Returns:
            Nom de l'action
        """
        actions = {
            0: "← Gauche",
            1: "○ Neutre",
            2: "→ Droite"
        }
        return actions.get(action, f"Action {action}")
    
    def close(self):
        """Ferme la fenêtre pygame"""
        pygame.quit()
