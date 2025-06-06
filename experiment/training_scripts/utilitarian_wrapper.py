# utilitarian_wrapper.py
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Dict
import numpy as np

# Import the original environment from mpe2
from mpe2 import simple_spread_v3 # This import assumes mpe2 is installed or accessible

class UtilitarianRewardWrapper(gym.Wrapper):
    """
    A wrapper for PettingZoo environments to apply a utilitarian reward function,
    where every agent receives the sum of all agents' individual rewards.
    """
    def __init__(self, env):
        super().__init__(env)
        # The observation and action spaces remain the same as the base environment
        # as we are only modifying the reward.

    def step(self, action):
        # Perform a step in the underlying environment
        observations, rewards, terminations, truncations, infos = self.env.step(action)

        # Calculate the total reward from all agents for the current step
        # 'rewards' is expected to be a dictionary mapping agent_id to individual reward
        total_step_reward = sum(rewards.values())

        # Create a new dictionary where each agent receives the total_step_reward
        utilitarian_rewards = {agent_id: total_step_reward for agent_id in rewards.keys()}

        # Return the original observations, the new utilitarian rewards, and other info
        return observations, utilitarian_rewards, terminations, truncations, infos

    # The reset method does not need to be modified as rewards are handled in step()
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)