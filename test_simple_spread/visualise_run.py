# visualise_simple_spread_data.py

import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def visualize_simple_spread_data(json_filename="../runs/test_simple_spread/runs/simple_spread_output_20250527_163837.json"):
    """
    Loads data from a specified JSON file, processes it, and visualizes
    the reward per agent over steps for each episode.

    Args:
        json_filename (str): The name of the JSON file containing the simulation data.
                             Assumes the file is in a 'runs' folder
                             one level up from the script's directory.
    """
    print(f"Starting data visualization script for {json_filename}.")

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the 'runs' directory by going up one level
    # from the script's directory and then into 'runs'
    prosocial_marl_dir = os.path.dirname(script_dir)
    runs_dir = os.path.join(prosocial_marl_dir, "runs") # Changed from "data" to "runs"
    json_filepath = os.path.join(script_dir, "runs", json_filename)

    # Create the 'plots' directory if it doesn't exist (assuming it's at the same level as 'runs' and 'scripts')
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        # Load the JSON data
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_filepath}")

        # Prepare data for plotting
        plot_data = []
        for episode_entry in data:
            episode_num = episode_entry["episode"]
            for step_entry in episode_entry["steps"]:
                plot_data.append({
                    "episode": episode_num,
                    "step_count": step_entry["step_count"],
                    "agent": step_entry["agent"],
                    "reward": step_entry["reward"]
                })

        # Convert to a pandas DataFrame for easier manipulation
        df = pd.DataFrame(plot_data)

        # Plotting the data
        plt.figure(figsize=(12, 8)) # Set figure size for better readability

        # Get unique agents and episodes for plotting
        agents = df["agent"].unique()
        episodes = df["episode"].unique()

        # Iterate through each episode and plot agent rewards
        for episode in episodes:
            episode_df = df[df["episode"] == episode]
            for agent in agents:
                agent_episode_df = episode_df[episode_df["agent"] == agent]
                plt.plot(agent_episode_df["step_count"], agent_episode_df["reward"], 
                         label=f"Episode {episode}, Agent {agent}", 
                         marker='o', linestyle='-') # Add markers for clarity

        plt.title("Reward per Agent Over Steps Across Episodes", fontsize=16)
        plt.xlabel("Step Count", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7) # Add a grid for easier reading
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) # Place legend outside plot
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # Save the plot
        plot_filename = f"simple_spread_rewards_{os.path.splitext(json_filename)[0]}.png"
        plot_filepath = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_filepath)
        print(f"Plot saved to {plot_filepath}")

    except FileNotFoundError:
        print(f"Error: The file '{json_filepath}' was not found.")
        print("Please ensure the JSON file is in the 'runs' folder relative to your script.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_filepath}'.")
        print("Please check if the JSON file is valid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("Data visualization script finished.")

# Call the function if running directly
if __name__ == "__main__":
    visualize_simple_spread_data("simple_spread_output_20250527_163010.json")