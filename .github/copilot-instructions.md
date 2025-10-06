# AI Agent Instructions for taxi-pbrl-project

Concise domain knowledge so an AI can contribute productively. Focus: Preference-based RL (PbRL) on Gymnasium Taxi-v3.

## 1. Architecture Overview
- Core paradigm: Extend classical tabular Q-Learning with human / simulated trajectory preferences.
- Main loop split into: (a) classical training (`QLearningAgent.train`), (b) preference application (`PreferenceBasedQLearning.update_from_preferences`), (c) interactive hybrid loop (`PreferenceBasedQLearning.interactive_training_loop`).
- Data objects: `TrajectoryStep` + `Trajectory` (dataclasses) collected by `TrajectoryManager.collect_trajectory` (evaluation mode: greedy actions, no exploration) to produce comparison material.
- Preference pipeline: `TrajectoryManager` -> pair selection (`_select_interesting_pairs`) -> user input via `PreferenceInterface` -> application to Q-table with shaped rewards (`update_from_preferences`).
- Persistence: Pickle for Q-tables / agents (`save_pbrl_agent`, classical variant in scripts), JSON for preference logs (`PreferenceInterface.save_preferences`), Pickle for raw trajectory lists (`TrajectoryManager.save_trajectories`).

## 2. Key Files
- `src/q_learning_agent.py`: Baseline tabular Q-Learning (ε-greedy, decay, train/evaluate, metrics arrays).
- `src/pbrl_agent.py`: Inherits QLearningAgent; adds preference-weighted reward shaping + interactive training orchestration.
- `src/trajectory_manager.py`: Episode collection (greedy), summarization, pairwise comparison display + visualization helpers.
- `src/preference_interface.py`: CLI-driven preference collection (batch & interactive) with lightweight reasoning capture.
- Scripts: `train_classical_agent.py`, `train_pbrl_agent.py`, `demo_preferences.py` drive workflows; `statistical_analysis.py` (not yet inspected here) likely aggregates results; `demo_preferences.py` showcases preference loop with pre-collected or live input.

## 3. Workflow Patterns
1. Create Gymnasium env: `env = gym.make("Taxi-v3")` (ensure `gymnasium[toy-text]` installed).
2. Classical training: instantiate `QLearningAgent(n_states, n_actions)` using `env.observation_space.n` & `env.action_space.n`, call `train(episodes=...)`, then optional `evaluate`.
3. Trajectory sampling (for preferences): use a *greedy* agent (set `epsilon=0` or rely on current ε) with `TrajectoryManager.collect_trajectory`.
4. Preference batch: feed pairs to `PreferenceInterface.collect_preference_batch` -> list[int] (0/1/2 choices).
5. Apply preferences: either initial bootstrapping (`_apply_existing_preferences`) or iterative inside `interactive_training_loop`.
6. Save artifacts to `results/` with consistent naming (`*_agent.pkl`, `demo_trajectories.pkl`, JSON summaries).

## 4. Conventions & Design Decisions
- State/action indices assumed integer-compatible with Taxi-v3 (discrete spaces). No wrappers used—keep code tabular (avoid introducing neural network layers unless explicitly requested).
- Reward shaping from preferences uses: bonus = strength * `preference_weight`; penalty = -0.5 * bonus. Updates apply reduced LR (50% of base) for stability.
- Later steps in a trajectory are down-weighted (linear 0→-30% scaling) when applying preference updates to emphasize early strategic choices.
- "Interesting" pair selection balances contrast (best vs worst) and subtle efficiency trade-offs (middle neighbors with efficiency delta > 0.1).
- Efficiency metric = total_reward / episode_length (guard division by zero). Success heuristic: total_reward > 0.
- CLI interaction uses simple blocking `input()`; keep additions consistent (avoid adding async/UI libs without need).

## 5. Extending Safely
When adding features:
- New reward shaping variants: add methods in `PreferenceBasedQLearning`—preserve existing public method signatures (`train_with_preferences`, `interactive_training_loop`).
- If adding logging/metrics, extend existing lists (`training_rewards`, `preference_learning_history`) rather than replacing; downstream scripts expect these names.
- For new persistence fields, append keys to dicts (maintain backward compatibility loading older pickles—add defensive defaulting if implementing load logic).
- Visualization additions: follow pattern in `TrajectoryManager.visualize_trajectories` (matplotlib, 2x2 grid) and save under `results/` with descriptive snake_case.

## 6. Dependencies & Environment
- Minimal runtime: `gymnasium`, `gymnasium[toy-text]`, `numpy`, `matplotlib` (optionally `pygame` if rendering in some configs). No deep RL libs required.
- Exploration parameter schedule: ε decays multiplicatively until `epsilon_min` (0.01 default). Do not silently change defaults—expose new knobs explicitly.

## 7. Common Pitfalls to Avoid
- Using `env.reset()` and `env.step()` signature must match Gymnasium (returns `(state, info)`; step returns 5-tuple including `terminated` & `truncated`). Keep compatibility.
- Do not mix training-time exploration with trajectory collection intended for preference comparison unless exploring deliberately—biases preference consistency.
- Ensure any added loops cap steps via `max_steps` to avoid infinite episodes (Taxi-v3 can loop).
- When adding statistical modules, reuse existing stored arrays instead of re-running expensive training unless necessary.

## 8. Example Snippets
Instantiate PbRL agent:
```python
agent = PreferenceBasedQLearning(env.observation_space.n, env.action_space.n, preference_weight=0.7)
```
Collect trajectories & preference:
```python
traj_manager = TrajectoryManager()
traj_a = traj_manager.collect_trajectory(env, agent)
traj_b = traj_manager.collect_trajectory(env, agent)
iface = PreferenceInterface()
choice = iface.collect_preference_interactive(traj_a, traj_b, traj_manager)
```
Apply preference manually:
```python
if choice == 1:
    agent.update_from_preferences(traj_a, traj_b)
elif choice == 2:
    agent.update_from_preferences(traj_b, traj_a)
```

## 9. Room for Future Enhancements (Only if asked)
- Bayesian preference modeling, active query selection, variance-aware exploration, batched render-less evaluation.

Keep responses focused on these patterns; avoid suggesting deep learning rewrites unless user explicitly pivots scope.
