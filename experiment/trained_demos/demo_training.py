import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from pettingzoo.mpe import simple_spread_v3
import numpy as np

# --- 1. Define Environment Parameters (MUST match training) ---
N_AGENTS = 3
LOCAL_RATIO = 0.5
MAX_CYCLES = 25

# --- 2. Define the exact env_creator function as used in training ---
def env_creator(args):
    """
    Creates and returns the Simple Spread environment instance.
    The 'args' (config) parameter will be passed by RLib, but often isn't directly used
    by the env_creator if parameters are fixed like N_AGENTS.
    """
    # Using render_mode="human" to visualize the agents
    return simple_spread_v3.env(N=N_AGENTS, local_ratio=LOCAL_RATIO, max_cycles=MAX_CYCLES, render_mode="human")

# --- 3. Register the environment with RLib (MUST match training's registered name) ---
# The lambda function wrapper is crucial for RLib to handle env creation with its config system
register_env("simple_spread_pz", lambda config: PettingZooEnv(env_creator(config)))

# --- 4. Configure the RLib Algorithm (MUST match training config) ---
# This config object defines the algorithm and the policy network.
# Crucially, it needs the environment name and the model architecture.
config = (
    PPOConfig()
    .environment(env="simple_spread_pz") # Use the registered env name
    .framework("torch") 
    .AlgorithmConfig.env_runners(num_envs_per_env_runner=0,
                num_envs_per_worker=1) # Set to 0 for local inference
    .training(
        model={
            "fcnet_hiddens": [64, 64], # Match your trained model architecture
            "fcnet_activation": "tanh", # Common default, specify if different (e.g., "relu")
                                        # If not specified in your original config, it's likely tanh
        }
    )
    # If you had any other specific config settings (e.g., learning rate, gamma, etc.)
    # you might add them here, but for inference, the model and env setup are most critical.
)

# --- 5. Define the Path to the Saved Checkpoint ---
# This path is relative to where you run this script, or an absolute path.
# IMPORTANT: This path should point to the PARENT directory of the actual checkpoint files.
# The checkpoint_000099 directory contains files like `policies/default_policy` and `checkpoint`.
CHECKPOINT_PATH = "experiment/run_data/simple_spread_baseline_standard_reward/PPO_simple_spread_pz_eeb92_00000_0_2025-06-04_16-19-31/checkpoint_000099"

# --- 6. Initialize Ray ---
# Start Ray if not already running.
if not ray.is_initialized():
    ray.init()

# --- 7. Create an RLib Trainer and Restore the Policy ---
print(f"Loading policy from checkpoint: {CHECKPOINT_PATH}")
# Trainer.restore() is the standard way to load a trained model in RLib
try:
    # `Trainer.restore` will automatically find the checkpoint files within the directory
    algo = config.build() # Build the algorithm with the defined configuration
    algo.restore(CHECKPOINT_PATH)
    print("RLlib Trainer and policy successfully restored!")
except Exception as e:
    print(f"Error restoring RLib Trainer: {e}")
    print("Please ensure the CHECKPOINT_PATH is correct and contains valid RLib checkpoint files.")
    print("Also, verify your RLib installation and version compatibility.")
    ray.shutdown()
    exit()

# --- 8. Evaluate the Policy ---
NUM_EVAL_EPISODES = 5
total_rewards_per_episode = []

print(f"\n--- Starting Evaluation of {NUM_EVAL_EPISODES} Episodes ---")

# Get a fresh environment instance for evaluation
# We use the registered name to get the environment as RLib would
env = PettingZooEnv(env_creator(None)) # Pass None for config if env_creator doesn't use it

for episode_num in range(NUM_EVAL_EPISODES):
    obs, info = env.reset() # Reset the environment to get initial observations
    episode_reward = {agent_id: 0.0 for agent_id in env.possible_agents}
    done = {"__all__": False} # RLib's done dictionary structure

    print(f"\n--- Episode {episode_num + 1} ---")

    while not done["__all__"]:
        actions = {}
        # Iterate over active agents to get their observations and compute actions
        for agent_id in env.agents: # env.agents contains only currently active agents
            # RLib's compute_single_action expects a single observation (not batched for single agent)
            # and returns a single action.
            # No need for manual batching/unbatching for individual agent steps here.
            
            # Ensure observation is in a format compatible with the policy (e.g., numpy array)
            # RLib usually handles internal conversion, but explicit conversion might be needed if issues arise.
            action, _, _ = algo.compute_single_action(
                observation=obs[agent_id],
                agent_id=agent_id,
                explore=False # Set to False for deterministic actions during evaluation
            )
            actions[agent_id] = action

        # Step the environment with actions from all active agents
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # Accumulate rewards
        for agent_id, r in rewards.items():
            episode_reward[agent_id] += r

        # Update observations for the next step
        obs = next_obs

        # Update global done status for the episode loop
        # RLib often uses a "__all__": True when all agents are done
        done = {"__all__": all(terminations.values()) or all(truncations.values())}

    # Episode ended
    total_episode_reward = sum(episode_reward.values())
    total_rewards_per_episode.append(total_episode_reward)
    print(f"Episode {episode_num + 1} finished. Total reward: {total_episode_reward:.2f}")

# --- 9. Final Cleanup ---
env.close() # Close the PettingZoo environment
ray.shutdown() # Shut down Ray

print(f"\n--- Evaluation Summary ---")
print(f"Average total reward over {NUM_EVAL_EPISODES} episodes: {np.mean(total_rewards_per_episode):.2f}")
print(f"Standard deviation of total rewards: {np.std(total_rewards_per_episode):.2f}")