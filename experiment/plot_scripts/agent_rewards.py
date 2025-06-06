import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import glob # For finding files recursively

def plot_learning_curves(file_path, y_min=None, y_max=None):
    """
    Reads a CSV file, identifies agent reward columns, plots individual learning curves
    for each agent, and saves each plot in a 'plots' folder, naming it based on the
    containing folder of the progress.csv file and the agent ID.

    Args:
        file_path (str): The path to the CSV file.
        y_min (float, optional): The minimum value for the y-axis. If None, matplotlib
                                 will automatically determine the lower limit.
        y_max (float, optional): The maximum value for the y-axis. If None, matplotlib
                                 will automatically determine the upper limit.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # --- Determine plot filename base and directory name ---
    csv_dir = os.path.dirname(file_path)
    if csv_dir == '':
        csv_dir_name = os.path.basename(os.getcwd())
    else:
        csv_dir_name = os.path.basename(csv_dir)

    plot_base_name_prefix = re.sub(r'[^\w\-_\. ]', '', csv_dir_name).strip()
    if not plot_base_name_prefix:
        plot_base_name_prefix = "learning_curve_from_current_folder"

    # --- Identify reward columns ---
    reward_columns = [col for col in df.columns if 'env_runners/agent_episode_returns_mean/agent_' in col]

    if not reward_columns:
        print(f"No agent-specific reward columns found in {file_path}.")
        return

    if 'training_iteration' not in df.columns:
        print(f"Error: 'training_iteration' column not found in {file_path} for the x-axis.")
        return

    # --- Saving the plots to the 'plots' folder ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_folder = os.path.abspath(os.path.join(script_dir, '..'))
    plots_folder = os.path.join(experiment_folder, 'plots')

    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        print(f"Created directory: '{plots_folder}'")

    # --- Plotting logic: Create a separate graph for each agent ---
    for col in reward_columns:
        match = re.search(r'agent_(\d+)', col)
        if match:
            agent_id = match.group(1)
            agent_name = f'Agent {agent_id}'

            plt.figure(figsize=(10, 6)) # Create a new figure for each agent
            plt.plot(df['training_iteration'], df[col], label=agent_name, color='blue', alpha=0.8) # You can pick a consistent color or cycle if you wish

            plt.xlabel('Training Iteration')
            plt.ylabel('Mean Episode Reward')
            plt.title(f'Learning Curve for {agent_name} in: {csv_dir_name}')
            plt.legend()
            plt.grid(True)

            # --- Set consistent y-axis limits (if provided) ---
            if y_min is not None and y_max is not None:
                plt.ylim(y_min, y_max)
            elif y_min is not None:
                plt.ylim(bottom=y_min)
            elif y_max is not None:
                plt.ylim(top=y_max)

            # Generate unique filename for each agent's plot
            plot_filename = f"{plot_base_name_prefix}_agent_{agent_id}_learning_curve.png"
            save_path = os.path.join(plots_folder, plot_filename)
            plt.savefig(save_path)
            print(f"Plot saved successfully to: '{save_path}'")

            plt.close() # Close the plot to free memory

# --- Main execution block for dynamic plotting ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_data_root = os.path.abspath(os.path.join(script_dir, '../run_data'))

    if not os.path.exists(run_data_root):
        print(f"Error: The 'run_data' directory was not found at '{run_data_root}'.")
        print("Please ensure your script is located correctly relative to 'run_data' or adjust the path.")
        exit()

    all_progress_files = glob.glob(os.path.join(run_data_root, '**', 'progress.csv'), recursive=True)

    if not all_progress_files:
        print(f"No 'progress.csv' files found in '{run_data_root}' or its subdirectories.")
        exit()

    print(f"Found {len(all_progress_files)} 'progress.csv' files.")

    global_min_reward = float('inf')
    global_max_reward = float('-inf')
    valid_files_for_range = []

    # First pass: Determine the global min and max reward across all relevant files
    print("Calculating global min and max reward across all files and all agents...")
    for file_path in all_progress_files:
        try:
            df = pd.read_csv(file_path)
            reward_columns = [col for col in df.columns if 'env_runners/agent_episode_returns_mean/agent_' in col]
            if reward_columns and 'training_iteration' in df.columns:
                valid_files_for_range.append(file_path)
                for col in reward_columns:
                    current_min = df[col].min()
                    current_max = df[col].max()
                    if pd.notna(current_min) and current_min < global_min_reward:
                        global_min_reward = current_min
                    if pd.notna(current_max) and current_max > global_max_reward:
                        global_max_reward = current_max
            else:
                print(f"Skipping {file_path}: Missing reward or training_iteration columns for range calculation.")
        except Exception as e:
            print(f"Could not read {file_path} for range calculation: {e}")

    if not valid_files_for_range:
        print("No valid files found to determine a global reward range. Exiting.")
        exit()

    # Add a small buffer to the min/max for better visualization
    buffer = (global_max_reward - global_min_reward) * 0.05
    final_y_min = global_min_reward - buffer
    final_y_max = global_max_reward + buffer

    print(f"\nDetermined global y-axis range: [{final_y_min:.2f}, {final_y_max:.2f}]")

    # Second pass: Plot all files using the determined global min and max
    print("\nGenerating separate plots for each agent with consistent y-axis...")
    for file_path in all_progress_files:
        plot_learning_curves(file_path, y_min=final_y_min, y_max=final_y_max)

    print("\nAll plots generated.")