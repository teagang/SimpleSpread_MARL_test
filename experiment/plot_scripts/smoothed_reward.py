import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import glob # For finding files recursively

def plot_learning_curves(file_path, y_min=None, y_max=None, window_size=10):
    """
    Reads a CSV file, identifies agent reward columns, plots learning curves,
    and saves the plot in a 'plots' folder, naming it based on the
    containing folder of the progress.csv file. Applies smoothing to the reward data.

    Args:
        file_path (str): The path to the CSV file.
        y_min (float, optional): The minimum value for the y-axis. If None, matplotlib
                                 will automatically determine the lower limit.
        y_max (float, optional): The maximum value for the y-axis. If None, matplotlib
                                 will automatically determine the upper limit.
        window_size (int, optional): The size of the rolling window for smoothing.
                                     Set to 1 for no smoothing. Defaults to 10.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # --- Determine plot filename based on the parent folder of the CSV ---
    csv_dir = os.path.dirname(file_path)
    if csv_dir == '':
        csv_dir_name = os.path.basename(os.getcwd())
    else:
        csv_dir_name = os.path.basename(csv_dir)

    plot_base_name = re.sub(r'[^\w\-_\. ]', '', csv_dir_name).strip()
    if not plot_base_name:
        plot_base_name = "learning_curve_from_current_folder"

    # Add smoothing info to filename
    plot_filename = f"{plot_base_name}_learning_curve_smoothed_ws{window_size}.png"

    # --- Plotting logic ---
    reward_columns = [col for col in df.columns if 'env_runners/agent_episode_returns_mean/agent_' in col]

    if not reward_columns:
        print(f"No agent-specific reward columns found in {file_path}.")
        return

    if 'training_iteration' not in df.columns:
        print(f"Error: 'training_iteration' column not found in {file_path} for the x-axis.")
        return

    plt.figure(figsize=(12, 7))
    for col in reward_columns:
        match = re.search(r'agent_(\d+)', col)
        if match:
            agent_id = match.group(1)
            agent_name = f'Agent {agent_id}'
            
            # Apply rolling mean for smoothing
            # .rolling(window=window_size).mean() calculates the rolling average.
            # .dropna() removes NaN values that appear at the beginning due to the rolling window.
            smoothed_rewards = df[col].rolling(window=window_size, min_periods=1).mean()
            
            plt.plot(df['training_iteration'], smoothed_rewards, label=agent_name, alpha=1.0)    

    plt.xlabel('Training Iteration')
    plt.ylabel(f'Mean Episode Reward (Smoothed with window={window_size})') # Update label
    plt.title(f'Learning Curves for: {csv_dir_name}')
    plt.legend()
    plt.grid(True)

    # --- Set consistent y-axis limits ---
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(bottom=y_min)
    elif y_max is not None:
        plt.ylim(top=y_max)

    # --- Saving the plot ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_folder = os.path.abspath(os.path.join(script_dir, '..'))
    plots_folder = os.path.join(experiment_folder, 'plots')

    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        print(f"Created directory: '{plots_folder}'")

    save_path = os.path.join(plots_folder, plot_filename)
    plt.savefig(save_path)
    print(f"Plot saved successfully to: '{save_path}'")

    plt.close() # Close the plot to free memory

# --- Main execution block for dynamic plotting ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming 'run_data' is in the parent directory of the script
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

    # Define a default smoothing window size for range calculation and plotting
    # You might want to experiment with this value.
    DEFAULT_SMOOTHING_WINDOW = 10 

    # First pass: Determine the global min and max reward across all relevant files
    # This part needs to consider the *smoothed* values for consistent y-axis
    print("Calculating global min and max reward across all files (considering smoothing)...")
    for file_path in all_progress_files:
        try:
            df = pd.read_csv(file_path)
            reward_columns = [col for col in df.columns if 'env_runners/agent_episode_returns_mean/agent_' in col]
            if reward_columns and 'training_iteration' in df.columns:
                valid_files_for_range.append(file_path)
                for col in reward_columns:
                    # Apply smoothing here for range calculation as well
                    smoothed_rewards = df[col].rolling(window=DEFAULT_SMOOTHING_WINDOW, min_periods=1).mean()
                    
                    current_min = smoothed_rewards.min()
                    current_max = smoothed_rewards.max()
                    
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
    print("\nGenerating plots with consistent y-axis and smoothing...")
    for file_path in all_progress_files:
        plot_learning_curves(file_path, y_min=final_y_min, y_max=final_y_max, window_size=DEFAULT_SMOOTHING_WINDOW)

    print("\nAll plots generated.")