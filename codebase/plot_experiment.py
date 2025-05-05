from data_analysis import plots, dataloading

import os
from argparse import ArgumentParser


def main(args):
    paths_to_data = (os.path.join(args.base_path_to_data_folders + str(experiment_index), "log") for experiment_index in range(*args.data_range))
    
    include_evaluation_fitness, include_fitness, include_novelty, include_runtime, include_time = args.include_evaluation_fitness, args.include_fitness, args.include_novelty, args.include_runtime, args.include_time
    if not include_evaluation_fitness and not include_fitness and not include_novelty and not include_runtime and not include_time:
        include_evaluation_fitness = True
        include_fitness = True
        include_novelty = True
        include_runtime = True
        include_time = True
    
    if args.experiment_type == "es":
        all_evaluation_fitnesses, all_fitnesses, all_runtimes, all_iteration_times = [], [], [], []
        
        print("Loading data...")
        for path in paths_to_data:
            evaluation_fitnesses, fitnesses, runtimes, iteration_times = dataloading.load_es_data(path, args.max_iterations)
            all_evaluation_fitnesses.append(evaluation_fitnesses)
            all_fitnesses.append(fitnesses)
            all_runtimes.append(runtimes)
            all_iteration_times.append(iteration_times)
            
        if include_evaluation_fitness:
            print("Plotting evaluation fitnesses...")
            plots.plot_evaluation_fitness(*all_evaluation_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions)
        
        if include_fitness:
            print("Plotting fitnesses...")
            plots.plot_fitness(*all_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions)
        
        if include_runtime:
            print("Plotting runtimes...")
            plots.plot_runtime(*all_runtimes, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions)
        
        if include_time:
            print("Plotting iteration wall-clock times...")
            plots.plot_time(*all_iteration_times, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions)
        
        
    elif args.experiment_type == "ns":
        all_evaluation_fitnesses, all_novelty_scores, all_runtimes, all_iteration_times = [], [], [], []
        
        print("Loading data...")
        for path in paths_to_data:
            evaluation_fitnesses, novelty_scores, runtimes, iteration_times = dataloading.load_ns_data(path, args.max_iterations)
            all_evaluation_fitnesses.append(evaluation_fitnesses)
            all_novelty_scores.append(novelty_scores)
            all_runtimes.append(runtimes)
            all_iteration_times.append(iteration_times)
            
        if include_evaluation_fitness:
            print("Plotting evaluation fitnesses...")
            plots.plot_evaluation_fitness(*all_evaluation_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions)
            
        if include_novelty:
            print("Plotting novelty scores...")
            plots.plot_novelty(*all_novelty_scores, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_novelty), plot_dimensions=args.plot_dimensions)
            
        if include_runtime:
            print("Plotting runtimes...")
            plots.plot_runtime(*all_runtimes, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions)
            
        if include_time:
            print("Plotting iteration wall-clock times...")
            plots.plot_time(*all_iteration_times, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions)
        
        
    elif args.experiment_type == "qd":
        all_evaluation_fitnesses, all_fitnesses, all_novelty_scores, all_runtimes, all_iteration_times = [], [], [], [], []
        
        print("Loading data...")
        for path in paths_to_data:
            evaluation_fitnesses, fitnesses, novelty_scores, runtimes, iteration_times = dataloading.load_qd_data(path, args.max_iterations)
            all_evaluation_fitnesses.append(evaluation_fitnesses)
            all_fitnesses.append(fitnesses)
            all_novelty_scores.append(novelty_scores)
            all_runtimes.append(runtimes)
            all_iteration_times.append(iteration_times)
            
        if include_evaluation_fitness:
            print("Plotting evaluation fitnesses...")
            plots.plot_evaluation_fitness(*all_evaluation_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions)
            
        if include_fitness:
            print("Plotting fitnesses...")
            plots.plot_fitness(*all_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions)
            
        if include_novelty:
            print("Plotting novelty scores...")
            plots.plot_novelty(*all_novelty_scores, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_novelty), plot_dimensions=args.plot_dimensions)
            
        if include_runtime:
            print("Plotting runtimes...")
            plots.plot_runtime(*all_runtimes, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions)
            
        if include_time:
            print("Plotting iteration wall-clock times...")
            plots.plot_time(*all_iteration_times, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions)
        


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("base_path_to_data_folders", type=str, help="Base path to the folders containing the logged data. Should be complete except for the variable part, so without the ending numbers.")
    parser.add_argument("-dr", "--data_range", type=int, nargs=2, default=(1, 11), help="Range of the numbers identifying the individual experiments (completing the paths to the experiment folders when appended to the 'base_path_to_data_folders'). (The lower bound is included, the upper bound is excluded, as is customary for ranges in Python.)")
    parser.add_argument("-t", "--experiment_type", type=str, default="es", help="Type of experiment. Either 'es', or 'ns', or 'qd'.")
    parser.add_argument("-i", "--max_iterations", type=int, default=200, help="Maximal number of iterations to be plotted.")
    parser.add_argument("-f", "--max_fitness", type=float, default=10, help="Maximal fitness value on the graphs.")
    parser.add_argument("-n", "--max_novelty", type=float, default=None, help="Maximal novelty value on the graphs.")
    parser.add_argument("-pd", "--plot_dimensions", nargs=2, type=float, default=(3.8, 2.7), help="Plot dimensions (two values, x and y).")
    parser.add_argument("-ief", "--include_evaluation_fitness", action="store_true", help="Plot the evaluation fitnesses. (If none such option is specified, all the data will be plotted.)")
    parser.add_argument("-if", "--include_fitness", action="store_true", help="Plot the fitnesses of the population. (If none such option is specified, all the data will be plotted.)")
    parser.add_argument("-in", "--include_novelty", action="store_true", help="Plot the novelty scores of the population. (If none such option is specified, all the data will be plotted.)")
    parser.add_argument("-ir", "--include_runtime", action="store_true", help="Plot the runtimes. (If none such option is specified, all the data will be plotted.)")
    parser.add_argument("-it", "--include_time", action="store_true", help="Plot the wall-clock time. (If none such option is specified, all the data will be plotted.)")
    
    main(parser.parse_args())
