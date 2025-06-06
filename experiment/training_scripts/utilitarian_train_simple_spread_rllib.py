import os
import ray
import gymnasium as gym # Added this import for clarity with gym.spaces
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from mpe2 import simple_spread_v3
from ray import tune

# Import your custom utilitarian reward wrapper
from utilitarian_wrapper import UtilitarianRewardWrapper #


# --- Configuration ---
TRAINING_ITERATIONS = 1000 # Number of training iterations/epochs
CHECKPOINT_FREQ = 100     # How often to save checkpoints
N_AGENTS = 3             # Number of agents in the environment
EXPERIMENT_NAME = "utilitarian_train_simple_spread_rllib.py"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_RUN_DATA_DIR = os.path.join(SCRIPT_DIR, "../run_data")
os.makedirs(OUTPUT_RUN_DATA_DIR, exist_ok=True)

# Initialize Ray
ray.init(ignore_reinit_error=True)

# --- 1. Environment Registration ---
def env_creator(args):
    # This defaults to discrete actions (Discrete(5)) if continuous_actions is not specified
    return simple_spread_v3.env(N=N_AGENTS, local_ratio=0.5, max_cycles=25)

# Register the environment with RLlib
register_env("simple_spread_pz_utilitarian", lambda config: PettingZooEnv(env_creator(config)))
print("PettingZoo environment registered with RLlib.")

# Define the policy mapping function
def policy_mapping_fn(agent_id, episode, **kwargs):
    return "shared_policy" # All agents will use the policy named "shared_policy"

# --- 2. RLlib PPO Configuration ---
# Create a single env instance for spaces
# This also defaults to discrete actions (Discrete(5))
single_env = simple_spread_v3.env(N=N_AGENTS, local_ratio=0.5, max_cycles=25)

config = (
    PPOConfig()
    .env_runners(num_env_runners=1, rollout_fragment_length=128)
    .training(
        gamma=0.99, # Discount factor for future rewards. O.99 is good for long term planning.
        lr=5e-4,
        kl_coeff=0.2,
        clip_param=0.1,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
        num_epochs=10,
        minibatch_size=128,
        train_batch_size=2048,
        model={"fcnet_hiddens": [64, 64]},
    )
    .environment(env="simple_spread_pz_utilitarian", clip_actions=False)
    .framework(framework="torch")
    .resources(num_gpus=0)
    .multi_agent(
        policies={
            "shared_policy": PolicySpec(
                None,
                single_env.observation_space("agent_0"),
                single_env.action_space("agent_0"), # This correctly passes Discrete(5) to the policy
                {},
            )
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["shared_policy"],
    )
)

print("RLlib PPO configuration created.")

# --- 3. Trainer Initialization and Training ---
print("Initializing RLlib PPO trainer...")
trainer = config.build()

print("Starting RLlib training...")
# Use tune.run to manage the experiment, logging, and checkpoints
results = tune.run(
    "PPO", # Algorithm to use
    name=EXPERIMENT_NAME, # Name of the experiment
    stop={"training_iteration": TRAINING_ITERATIONS}, # Stop condition
    config=config.to_dict(), # Pass the PPOConfig as a dictionary
    storage_path=OUTPUT_RUN_DATA_DIR,    # The `progress.csv` file is automatically saved in each trial's folder
    # e.g., ~/ray_results/simple_spread_baseline/PPO_simple_spread_pz_1a2b3_00000_0_.../progress.csv
    checkpoint_freq=10, # How often to save checkpoints (using tune's functionality)
    checkpoint_at_end=True, # Save a checkpoint at the end of training
    # verbose=1 # Adjust verbosity level (0: silent, 1: status, 2: detailed)
)

print("Training finished.")

# Accessing results
if results.trials:
    # Get the path to the latest trial's results
    latest_trial_logdir = results.trials[-1].local_path
    print(f"Latest trial results are in: {latest_trial_logdir}")

    # You can also access the last reported metrics for the latest trial
    last_metrics = results.trials[-1].last_result
    print(f"Last reported metrics: {last_metrics}")
    print(f"Mean reward: {last_metrics.get('episode_reward_mean')}")

ray.shutdown()