import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_average_rewards(run_data_dir, plots_dir, smoothing_window=20):
    """
    Generates a single graph showing the smoothed average episode returns for
    different training runs, searching recursively for progress.csv files.
    Legend is placed inside the plot.

    Args:
        run_data_dir (str): The path to the directory containing experiment runs.
                            progress.csv files can be in subfolders.
        plots_dir (str): The path to the directory where the generated plot
                         should be saved.
        smoothing_window (int): The window size for the rolling mean to smooth
                                plot lines. Set to 1 for no smoothing.
    """
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(14, 9)) # Keep the figure size
    plt.xlabel("Training Iteration", fontsize=12)
    plt.ylabel(f"Average Episode Return (Smoothed, Window={smoothing_window if smoothing_window > 1 else 'Raw'})", fontsize=12)
    plt.title("Smoothed Average Episode Returns for Different Training Runs", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.style.use('seaborn-v0_8-darkgrid')

    found_data_to_plot = False

    for root, dirs, files in os.walk(run_data_dir):
        if "progress.csv" in files:
            progress_file_path = os.path.join(root, "progress.csv")
            label_name = os.path.basename(root)

            try:
                df = pd.read_csv(progress_file_path)
                print(f"Processing: {progress_file_path} for run '{label_name}'")

                training_iteration_col = 'training_iteration'
                episode_return_mean_col = 'env_runners/episode_return_mean'

                if training_iteration_col in df.columns and episode_return_mean_col in df.columns:
                    if not df[episode_return_mean_col].dropna().empty:
                        if smoothing_window > 1:
                            plot_data = df[episode_return_mean_col].rolling(window=smoothing_window, center=True, min_periods=1).mean()
                        else:
                            plot_data = df[episode_return_mean_col]

                        plt.plot(df[training_iteration_col], plot_data, label=label_name, linewidth=2)
                        found_data_to_plot = True
                    else:
                        print(f"Info: Skipping '{label_name}'. No data in '{episode_return_mean_col}' in '{progress_file_path}'.")
                else:
                    print(f"Warning: Skipping '{label_name}'. Missing expected columns "
                          f"('{training_iteration_col}' or '{episode_return_mean_col}') in '{progress_file_path}'.")
            except pd.errors.EmptyDataError:
                print(f"Warning: Skipping '{label_name}'. '{progress_file_path}' is empty.")
            except Exception as e:
                print(f"Error processing '{progress_file_path}' for run '{label_name}': {e}")
    
    if not found_data_to_plot:
        print("\nWarning: No data was plotted. Ensure 'progress.csv' files exist and contain the required data.")
    else:
        # Place legend inside the plot. 'best' tries to find the least obstructive location.
        # Other options: 'upper left', 'upper right', 'lower left', 'lower right'
        plt.legend(title="Training Runs", loc='best', fontsize=10) # Added fontsize for legend text
        
        # Use tight_layout without the rect argument to allow the plot to fill the figure
        plt.tight_layout() 

    plot_filename = f"combined_average_rewards_smoothed_{smoothing_window if smoothing_window > 1 else 'raw'}_legend_inside.png"
    plot_save_path = os.path.join(plots_dir, plot_filename)

    plt.savefig(plot_save_path, dpi=300)
    print(f"\nSuccessfully generated and saved the combined plot to: {plot_save_path}")
    plt.close()

# --- How to use the script ---
if __name__ == "__main__":
    my_run_data_directory = "/Users/tegan/Documents/GitHub/prosocial_marl/experiment/run_data"
    my_plots_directory = "/Users/tegan/Documents/GitHub/prosocial_marl/experiment/plots"

    plot_average_rewards(my_run_data_directory, my_plots_directory, smoothing_window=20)
    print("\nScript execution complete. Check the specified plots directory for the output.")