from data_analysis import plots, dataloading

import os
from argparse import ArgumentParser


def main(args):
    all_evaluation_fitnesses, all_fitnesses, all_runtimes, all_iteration_times = [], [], [], []
    
    paths_to_data = (os.path.join(args.base_path_to_data_folders + str(experiment_index), "log") for experiment_index in range(*args.data_range))
    
    print("Loading data...")
    for path in paths_to_data:
        evaluation_fitnesses, fitnesses, runtimes, iteration_times = dataloading.load_es_data(path, args.max_iterations)
        all_evaluation_fitnesses.append(evaluation_fitnesses)
        all_fitnesses.append(fitnesses)
        all_runtimes.append(runtimes)
        all_iteration_times.append(iteration_times)
        
    print("Plotting evaluation fitnesses...")
    plots.plot_evaluation_fitness(*all_evaluation_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions)
    print("Plotting fitnesses...")
    plots.plot_fitness(*all_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions)
    print("Plotting runtimes...")
    plots.plot_runtime(*all_runtimes, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions)
    print("Plotting iteration wall-clock times...")
    plots.plot_time(*all_iteration_times, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("base_path_to_data_folders", type=str, help="Base path to the folders containing the logged data. Should be complete except for the variable part, so without the ending numbers.")
    parser.add_argument("-dr", "--data_range", type=int, nargs=2, default=(1, 11), help="Range of the numbers identifying the individual experiments (completing the paths to the experiment folders when appended to the 'base_path_to_data_folders'). (The lower bound is included, the upper bound is excluded, as is customary for ranges in Python.)")
    parser.add_argument("-i", "--max_iterations", type=int, default=200, help="Maximal number of iterations to be plotted.")
    parser.add_argument("-f", "--max_fitness", type=float, default=10, help="Maximal fitness value on the graphs.")
    parser.add_argument("-pd", "--plot_dimensions", nargs=2, type=float, default=(3.8, 2.7), help="Plot dimensions (two values, x and y).")
    
    main(parser.parse_args())
