"""
Module de discrétisation pour l'environnement MountainCar-v0
Convertit l'espace d'états continu en espace discret pour Q-Learning
"""

import numpy as np
from typing import Tuple, List


class MountainCarDiscretizer:
    """
    Discrétise l'espace d'états continu de MountainCar pour Q-Learning
    
    MountainCar State Space:
    - Position: [-1.2, 0.6] (0.5 = goal)
    - Vitesse: [-0.07, 0.07]
    """
    
    def __init__(self, n_position_bins: int = 20, n_velocity_bins: int = 20):
        """
        Initialise le discrétiseur
        
        Args:
            n_position_bins: Nombre de bins pour la position
            n_velocity_bins: Nombre de bins pour la vitesse
        """
        self.n_position_bins = n_position_bins
        self.n_velocity_bins = n_velocity_bins
        
        # Limites de l'environnement MountainCar
        self.position_min = -1.2
        self.position_max = 0.6
        self.velocity_min = -0.07
        self.velocity_max = 0.07
        
        # Création des bins
        self.position_bins = np.linspace(
            self.position_min, 
            self.position_max, 
            n_position_bins - 1
        )
        
        self.velocity_bins = np.linspace(
            self.velocity_min,
            self.velocity_max,
            n_velocity_bins - 1
        )
        
        # Nombre total d'états discrets
        self.n_states = n_position_bins * n_velocity_bins
        
        print(f"✅ MountainCarDiscretizer initialisé:")
        print(f"   - Bins position: {n_position_bins}")
        print(f"   - Bins vitesse: {n_velocity_bins}")
        print(f"   - Total états discrets: {self.n_states}")
    
    def discretize(self, state: np.ndarray) -> int:
        """
        Convertit un état continu en état discret
        
        Args:
            state: [position, velocity] array de taille 2
            
        Returns:
            État discret (index unique)
        """
        position, velocity = state
        
        # Discrétisation de chaque dimension
        pos_idx = np.digitize(position, self.position_bins)
        vel_idx = np.digitize(velocity, self.velocity_bins)
        
        # Conversion en index unique
        discrete_state = pos_idx * self.n_velocity_bins + vel_idx
        
        return discrete_state
    
    def continuous_to_discrete(self, state: np.ndarray) -> int:
        """Alias pour discretize()"""
        return self.discretize(state)
    
    def discrete_to_continuous(self, discrete_state: int) -> Tuple[float, float]:
        """
        Convertit un état discret en représentation continue (centre du bin)
        
        Args:
            discrete_state: Index de l'état discret
            
        Returns:
            Tuple (position, velocity) représentant le centre du bin
        """
        pos_idx = discrete_state // self.n_velocity_bins
        vel_idx = discrete_state % self.n_velocity_bins
        
        # Position centrale de chaque bin
        if pos_idx == 0:
            position = self.position_min
        elif pos_idx >= len(self.position_bins):
            position = self.position_max
        else:
            position = (self.position_bins[pos_idx - 1] + self.position_bins[pos_idx]) / 2
        
        if vel_idx == 0:
            velocity = self.velocity_min
        elif vel_idx >= len(self.velocity_bins):
            velocity = self.velocity_max
        else:
            velocity = (self.velocity_bins[vel_idx - 1] + self.velocity_bins[vel_idx]) / 2
        
        return position, velocity
    
    def get_bin_indices(self, state: np.ndarray) -> Tuple[int, int]:
        """
        Retourne les indices des bins pour chaque dimension
        
        Args:
            state: [position, velocity] array
            
        Returns:
            Tuple (pos_idx, vel_idx)
        """
        position, velocity = state
        
        pos_idx = np.digitize(position, self.position_bins)
        vel_idx = np.digitize(velocity, self.velocity_bins)
        
        return pos_idx, vel_idx
    
    def get_state_info(self, state: np.ndarray) -> dict:
        """
        Retourne des informations détaillées sur un état
        
        Args:
            state: [position, velocity] array
            
        Returns:
            Dictionnaire avec informations sur l'état
        """
        position, velocity = state
        discrete_state = self.discretize(state)
        pos_idx, vel_idx = self.get_bin_indices(state)
        
        # Calcul du progrès vers le but
        progress = (position - self.position_min) / (self.position_max - self.position_min)
        
        # Direction du mouvement
        if velocity > 0.01:
            direction = "→ Droite"
        elif velocity < -0.01:
            direction = "← Gauche"
        else:
            direction = "⊙ Stationnaire"
        
        return {
            'continuous_state': (position, velocity),
            'discrete_state': discrete_state,
            'position_bin': pos_idx,
            'velocity_bin': vel_idx,
            'progress_percent': progress * 100,
            'direction': direction,
            'near_goal': position >= 0.5
        }
    
    def visualize_discretization(self) -> str:
        """
        Crée une visualisation textuelle de la discrétisation
        
        Returns:
            String représentant la grille de discrétisation
        """
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"GRILLE DE DISCRÉTISATION MOUNTAINCAR")
        lines.append(f"{'='*60}")
        lines.append(f"Position: [{self.position_min:.2f}, {self.position_max:.2f}] → {self.n_position_bins} bins")
        lines.append(f"Vitesse:  [{self.velocity_min:.3f}, {self.velocity_max:.3f}] → {self.n_velocity_bins} bins")
        lines.append(f"Total états: {self.n_states}")
        lines.append(f"{'='*60}\n")
        
        return '\n'.join(lines)
    
    def get_statistics(self) -> dict:
        """
        Retourne des statistiques sur la discrétisation
        
        Returns:
            Dictionnaire de statistiques
        """
        position_resolution = (self.position_max - self.position_min) / self.n_position_bins
        velocity_resolution = (self.velocity_max - self.velocity_min) / self.n_velocity_bins
        
        return {
            'n_states': self.n_states,
            'n_position_bins': self.n_position_bins,
            'n_velocity_bins': self.n_velocity_bins,
            'position_resolution': position_resolution,
            'velocity_resolution': velocity_resolution,
            'position_range': (self.position_min, self.position_max),
            'velocity_range': (self.velocity_min, self.velocity_max)
        }


def test_discretizer():
    """Test du discrétiseur"""
    print("🧪 TEST DU DISCRÉTISEUR MOUNTAINCAR\n")
    
    discretizer = MountainCarDiscretizer(n_position_bins=20, n_velocity_bins=20)
    
    # Test de quelques états caractéristiques
    test_states = [
        ([-1.2, 0.0], "Position initiale (vallée gauche)"),
        ([-0.5, 0.0], "Centre de la vallée"),
        ([0.5, 0.0], "But (sommet droit)"),
        ([-0.5, 0.07], "Vitesse max droite"),
        ([-0.5, -0.07], "Vitesse max gauche"),
        ([0.0, 0.03], "Montée avec élan"),
    ]
    
    print(f"\n{'='*80}")
    print("TESTS D'ÉTATS CARACTÉRISTIQUES")
    print(f"{'='*80}\n")
    
    for state, description in test_states:
        state_array = np.array(state)
        info = discretizer.get_state_info(state_array)
        
        print(f"📍 {description}")
        print(f"   État continu: position={state[0]:.3f}, vitesse={state[1]:.3f}")
        print(f"   État discret: {info['discrete_state']}")
        print(f"   Bins: position={info['position_bin']}, vitesse={info['velocity_bin']}")
        print(f"   Progrès: {info['progress_percent']:.1f}%")
        print(f"   Direction: {info['direction']}")
        print(f"   But atteint: {'✅' if info['near_goal'] else '❌'}")
        print()
    
    # Test de la conversion inverse
    print(f"\n{'='*80}")
    print("TEST CONVERSION INVERSE (discret → continu)")
    print(f"{'='*80}\n")
    
    discrete_state = 150
    continuous = discretizer.discrete_to_continuous(discrete_state)
    print(f"État discret {discrete_state} → continu: position={continuous[0]:.3f}, vitesse={continuous[1]:.3f}")
    
    # Test round-trip
    original_state = np.array([-0.3, 0.02])
    discrete = discretizer.discretize(original_state)
    recovered = discretizer.discrete_to_continuous(discrete)
    print(f"\nRound-trip:")
    print(f"  Original: {original_state}")
    print(f"  Discret: {discrete}")
    print(f"  Récupéré: position={recovered[0]:.3f}, vitesse={recovered[1]:.3f}")
    
    # Statistiques
    print(f"\n{'='*80}")
    print("STATISTIQUES")
    print(f"{'='*80}\n")
    
    stats = discretizer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(discretizer.visualize_discretization())
    
    print("✅ Tests terminés!")


if __name__ == "__main__":
    test_discretizer()
