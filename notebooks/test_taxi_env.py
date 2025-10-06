import gymnasium as gym

# Création et test de l'environnement Taxi-v3
env = gym.make("Taxi-v3", render_mode="human")

# Test de base : reset et un pas
obs, info = env.reset()
print("Observation initiale:", obs)

obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(f"Après une action aléatoire: obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}")

env.close()