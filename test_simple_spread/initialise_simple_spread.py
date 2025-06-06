# Import the multi-agent particle environment from PettingZoo
from pettingzoo.mpe import simple_spread_v3
import numpy as np
import time

def run_simple_spread_aec_script():
    print("Starting AEC version of simple_spread script.")
    """
    Initializes the simple_spread environment with AEC API and runs for a short duration.
    AEC = Agent-Environment-Cycle: agents act one at a time in sequence
    This is opposed to parallel execution where all agents act simultaneously
    """
    try:
         # Create the simple_spread environment with specific parameters:
        # N=3: Number of agents and landmarks (3 agents must cover 3 landmarks)
        # local_ratio=0.5: How much agents are rewarded for individual vs collective success
        # max_cycles=25: Maximum number of steps per episode before automatic termination
        # render_mode="human": Display the environment visually in a window
        # continuous_actions=True: Agents use continuous movement (not discrete steps)
        env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25, render_mode="human", continuous_actions=True)
        print("Environment created successfully.")
        # Reset the environment to initial state - this must be called before using the environment
        # Reset initializes agent positions, landmark positions, and clears any previous state
        env.reset()
        print("Environment reset successfully.")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        print("Make sure you have pettingzoo installed: pip install pettingzoo[mpe]")
        return # Exit the function if environment creation fails

    print("Environment initialized with rendering.")
    print(f"Agents in environment: {env.agents}")
    
    # Display the observation and action spaces for the first agent
    # All agents in simple_spread have identical spaces, so we only need to check one
    if env.agents:
        first_agent = env.agents[0]
        # Observation space: what information each agent receives about the environment
        # In simple_spread: agent position, velocity, landmark positions, other agent positions (if visible)
        print(f"Observation space for {first_agent}: {env.observation_space(first_agent)}")
        # Action space: what actions each agent can take
        # In simple_spread with continuous_actions=True: [force_x, force_y] - continuous movement forces
        print(f"Action space for {first_agent}: {env.action_space(first_agent)}")
    # An episode is one complete game from start to finish (either max_cycles reached or all agents done)
    num_episodes = 3  

    # Main simulation loop - run multiple episodes
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1} ===")
        # Reset environment at the start of each new episode
        # This randomizes agent and landmark positions for a fresh start
        env.reset()
        
        step_count = 0 # Track how many individual agent actions we've taken
        # AEC API: iterate through agents one by one until episode ends
        # env.agent_iter() yields the next agent that needs to take an action
        # This continues until all agents are done or max_cycles is reached
        for agent in env.agent_iter():
            step_count += 1
            # Get the current state information for this agent
            # observation: what the agent can see (positions, velocities, etc.)
            # reward: how much reward the agent received from the last action
            # termination: True if the agent is permanently done (task completed)
            # truncation: True if the agent is done due to time limit (max_cycles)
            # info: additional debug information (usually empty)
            observation, reward, termination, truncation, info = env.last()
            
            print(f"Step {step_count}, Agent {agent}")
            print(f"  Reward: {reward:.3f}, Term: {termination}, Trunc: {truncation}")
            
            if termination or truncation:
                action = None
                print(f"  Agent {agent} is done, no action taken")
            else:
                action = env.action_space(agent).sample()
                print(f"  Agent {agent} taking action: {action}")

            # Apply the action to the environment
            # This updates the agent's position and triggers physics simulation
            # The environment then moves to the next agent or ends the episode
            env.step(action)
            
            # Add a small delay for visualization
            time.sleep(0.05)
            
            # Break if all agents are done (episode finished)
            if env.agents == []:
                print(f"Episode {episode + 1} finished after {step_count} steps")
                break
    # Clean up: close the environment window and free resources
    try:
        env.close()
        print("\nEnvironment closed successfully.")
    except Exception as e:
        print(f"Error closing environment: {e}")
        
    print("Script finished.")

# Call the function if running directly
if __name__ == "__main__":
    run_simple_spread_aec_script()