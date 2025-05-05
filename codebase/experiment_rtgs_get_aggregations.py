from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser
import math

import numpy as np


def main(args):
    experiment_folder = Path(args.path_to_experiment_folder)
    run_folders = [f for f in experiment_folder.iterdir() if f.is_dir()]
    rtg_dependent_returns, rtg_dependent_lengths = defaultdict(list), defaultdict(list)
    rtg_dependent_aggregated_returns, rtg_dependent_aggregated_lengths = defaultdict(list), defaultdict(list)
    
    for run_folder in run_folders:
        performance_folder = Path.joinpath(run_folder, "ckpts", "performance")
        
        # Data for mean and standard deviations
        # All the non-aggregated rtg files in the run folder
        # These files contain the performance of the model for a specific rtg value
        different_rtg_files = [f for f in performance_folder.iterdir() if f.is_file() and f.name.endswith(".csv") and not f.name.endswith("_aggregated.csv")]
        
        for file in different_rtg_files:
            desired_rtg = float(file.stem)
            
            with open(file, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Skip the header line
                    for line in lines[1:]:
                        episode_return, episode_length = line.strip().split(";")
                        
                        rtg_dependent_returns[desired_rtg].append(float(episode_return))
                        rtg_dependent_lengths[desired_rtg].append(int(episode_length))
        
        # Data for median and quartiles
        # All the aggregated rtg files in the run folder
        # These files contain the aggregated performance of the model for a specific rtg value
        different_rtg_aggregated_files = [f for f in performance_folder.iterdir() if f.is_file() and f.name.endswith("_aggregated.csv")]
        
        for file in different_rtg_aggregated_files:
            desired_rtg = float(file.stem.split("_")[0])
            
            with open(file, "r") as f:
                lines = f.readlines()
                
                mean_episode_return, _, mean_episode_length, _ = lines[1].strip().split(";")
                        
                rtg_dependent_aggregated_returns[desired_rtg].append(float(episode_return))
                rtg_dependent_aggregated_lengths[desired_rtg].append(int(episode_length))
    
    for rtg in rtg_dependent_returns:
        # Means and standard deviations
        mean_returns = sum(rtg_dependent_returns[rtg]) / len(rtg_dependent_returns[rtg])
        mean_lengths = sum(rtg_dependent_lengths[rtg]) / len(rtg_dependent_lengths[rtg])
        std_returns = math.sqrt(sum((x - mean_returns) ** 2 for x in rtg_dependent_returns[rtg]) / len(rtg_dependent_returns[rtg]))
        std_lengths = math.sqrt(sum((x - mean_lengths) ** 2 for x in rtg_dependent_lengths[rtg]) / len(rtg_dependent_lengths[rtg]))
        
        aggregated_file_path = experiment_folder / f"{rtg}_mean_and_std.csv"
        with open(aggregated_file_path, "w") as f:
            f.write("Mean Return;Std Return;Mean Length;Std Length\n")
            f.write(f"{mean_returns};{std_returns};{mean_lengths};{std_lengths}\n")
            
        # Median and quartiles
        mean_return_quartiles = np.quantile(rtg_dependent_returns[rtg], [0.25, 0.5, 0.75])
        mean_length_quartiles = np.quantile(rtg_dependent_lengths[rtg], [0.25, 0.5, 0.75])
        
        aggregated_file_path = experiment_folder / f"{rtg}_quartiles.csv"
        with open(aggregated_file_path, "w") as f:
            f.write("Q1 Return;Median Return;Q3 Return;Q1 Length;Median Length;Q3 Length\n")
            f.write(f"{mean_return_quartiles[0]};{mean_return_quartiles[1]};{mean_return_quartiles[2]};{mean_length_quartiles[0]};{mean_length_quartiles[1]};{mean_length_quartiles[2]}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("path_to_experiment_folder", type=str, help="Path to the experiment folder containing the directories of individual runs of the experiment with logged data of various rtgs runs to be processed.")
        
    main(parser.parse_args())
