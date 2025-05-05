from data_analysis import plots, dataloading

from pathlib import Path
from argparse import ArgumentParser


def main(args):
    if args.experiment_names is not None:
        if len(args.experiment_names) != len(args.paths_to_experiment_folders):
            raise ValueError("The number of experiment names must match the number of base paths to the data folders.")
        experiment_names = args.experiment_names
    else:
        experiment_names = [str(i+1) for i in range(len(args.paths_to_experiment_folders))]
        
    if args.central_measure not in ["mean", "median"]:
        raise ValueError("Invalid central measure argument. Possible argument values are 'mean' and 'median'.")
    
    if args.add_line is not None:
        line_name, y_value = args.add_line
        try:
            y_value = float(y_value)
        except ValueError:
            raise ValueError("The y-value of the horizontal dashed line must be a float.")
    else:
        line_name, y_value = None, None
        
    values_range = (0, None)
    
    print("Creating the plot...")
    plots.create_plot_for_multiple_experiments(
        plot_dimensions=args.plot_dimensions
    )

    match args.plot_type:
        case "eval":
            for i, experiment_path in enumerate(args.paths_to_experiment_folders):
                path_to_experiment_folder = Path(experiment_path)
                current_experiment_evaluation_data = []

                if args.experiment_names is not None:
                    print(f"Processing data from experiment named {experiment_names[i]}...")

                else:
                    print(f"Processing data from experiment number {i+1}:")

                print("Loading data...")
                paths_to_evaluation_data = [run_directory / "log.evaluations.csv" for run_directory in path_to_experiment_folder.iterdir() if run_directory.is_dir()]

                single_run_identifiers = list()
                for path in paths_to_evaluation_data:
                    print("Loading data from file:", path)
                    new_evaluation_dataframe = dataloading.load_evaluation_fitnesses_from_csv(path, args.max_iterations)
                    current_experiment_evaluation_data.append(new_evaluation_dataframe)
                    single_run_identifiers.append(path.parent.stem)

                print("Adding evaluation fitnesses to the plot...")
                plots.add_evaluation_data_from_one_experiment_to_plot(
                    *current_experiment_evaluation_data,
                    experiment_name=experiment_names[i],
                    single_run_identifiers=single_run_identifiers,
                    central_measure=args.central_measure,
                    interval_measure=args.interval_measure
                )
            
            values_range=(0, args.max_fitness)
            
        case "fitness":
            for i, experiment_path in enumerate(args.paths_to_experiment_folders):
                path_to_experiment_folder = Path(experiment_path)
                current_experiment_fitness_data = []

                if args.experiment_names is not None:
                    print(f"Processing data from experiment named {experiment_names[i]}...")

                else:
                    print(f"Processing data from experiment number {i+1}:")

                print("Loading data...")
                paths_to_fitness_data = [run_directory / "log.fitness.csv" for run_directory in path_to_experiment_folder.iterdir() if run_directory.is_dir()]

                single_run_identifiers = list()
                for path in paths_to_fitness_data:
                    print("Loading data from file:", path)
                    new_fitness_dataframe = dataloading.load_fitnesses_from_csv(path, args.max_iterations)
                    current_experiment_fitness_data.append(new_fitness_dataframe)
                    single_run_identifiers.append(path.parent.stem)

                print("Adding population fitnesses to the plot...")
                plots.add_fitness_data_from_one_experiment_to_plot(
                    *current_experiment_fitness_data,
                    experiment_name=experiment_names[i],
                    single_run_identifiers=single_run_identifiers,
                    central_measure=args.central_measure,
                    interval_measure=args.interval_measure
                )
                
                values_range=(0, args.max_fitness)
        
        case "runtime":
            for i, experiment_path in enumerate(args.paths_to_experiment_folders):
                path_to_experiment_folder = Path(experiment_path)
                current_experiment_runtime_data = []

                if args.experiment_names is not None:
                    print(f"Processing data from experiment named {experiment_names[i]}...")

                else:
                    print(f"Processing data from experiment number {i+1}:")

                print("Loading data...")
                paths_to_runtime_data = [run_directory / "log.runtime.csv" for run_directory in path_to_experiment_folder.iterdir() if run_directory.is_dir()]

                single_run_identifiers = list()
                for path in paths_to_runtime_data:
                    print("Loading data from file:", path)
                    new_runtime_dataframe = dataloading.load_runtimes_from_csv(path, args.max_iterations)
                    current_experiment_runtime_data.append(new_runtime_dataframe)
                    single_run_identifiers.append(path.parent.stem)

                print("Adding runtimes to the plot...")
                plots.add_runtime_data_from_one_experiment_to_plot(
                    *current_experiment_runtime_data,
                    experiment_name=experiment_names[i],
                    single_run_identifiers=single_run_identifiers,
                    central_measure=args.central_measure,
                    interval_measure=args.interval_measure
                )
            
        case "time":
            for i, experiment_path in enumerate(args.paths_to_experiment_folders):
                path_to_experiment_folder = Path(experiment_path)
                current_experiment_time_data = []

                if args.experiment_names is not None:
                    print(f"Processing data from experiment named {experiment_names[i]}...")

                else:
                    print(f"Processing data from experiment number {i+1}:")

                print("Loading data...")
                paths_to_time_data = [run_directory / "log.time.csv" for run_directory in path_to_experiment_folder.iterdir() if run_directory.is_dir()]

                single_run_identifiers = list()
                for path in paths_to_time_data:
                    print("Loading data from file:", path)
                    new_time_dataframe = dataloading.load_times_from_csv(path, args.max_iterations)
                    current_experiment_time_data.append(new_time_dataframe)
                    single_run_identifiers.append(path.parent.stem)

                print("Adding wall-clock times to the plot...")
                plots.add_time_data_from_one_experiment_to_plot(
                    *current_experiment_time_data,
                    experiment_name=experiment_names[i],
                    single_run_identifiers=single_run_identifiers,
                    central_measure=args.central_measure,
                    interval_measure=args.interval_measure
                )
    
        case _:
            raise ValueError("Invalid plot type argument. Possible argument values are 'eval' for evaluation fitnesses, 'fitness' for population fitnesses, 'runtime' for runtimes, and 'time' for wall-clock time.")

    if y_value is not None:
        print(f"Adding a horizontal dashed line with name '{line_name}' and y-value {y_value} to the plot...")
        plots.add_horizontal_dashed_line_to_plot(line_name, y_value)

    print("Rendering the plot...")
    plots.show_plot_for_multiple_experiments(
        num_of_iterations_to_plot=args.max_iterations,
        values_range=values_range,
        disable_legend=(args.experiment_names is None)
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("paths_to_experiment_folders", type=str, nargs="+", help="Path(s) to the experiment folder(s) containing the directories of individual runs of the experiment with logged data to be plotted.")
    parser.add_argument("-t", "--plot_type", type=str, default="eval", help="Type of the data to plot from the experiments. Possible values are 'eval' for evaluation fitnesses, 'fitness' for population fitnesses, 'runtime' for runtimes, and 'time' for wall-clock time.")
    parser.add_argument("-i", "--max_iterations", type=int, default=200, help="Maximal number of iterations to be plotted.")
    parser.add_argument("-f", "--max_fitness", type=float, default=10, help="Maximal fitness value on the graphs.")
    parser.add_argument("-pd", "--plot_dimensions", nargs=2, type=float, default=(3.8, 2.7), help="Plot dimensions (two values, x and y).")
    parser.add_argument("-n", "--experiment_names", type=str, nargs="+", help="Names for the individual experiments in the order as passed to the base path argument.")
    parser.add_argument("-cm", "--central_measure", type=str, default="median", help="Central measure to be plotted. Possible values are 'mean' and 'median'.")
    parser.add_argument("-im", "--interval_measure", type=str, default="quartiles", help="Interval measure to be plotted. Possible values are 'quartiles' and 'standard', which stands for standard percentile interval (the whole interval for one-value-per-iteration data and 95% interval for multiple-values-per-iteration data).")
    parser.add_argument("-l", "--add_line", nargs=2, type=str, metavar=("NAME", "Y_VALUE"), help="Add a horizontal dashed line with the given name (string) and y-value (float) to the plot. (The name of the line will be shown only when the experiments are named as well, otherwise the legend will be disabled.)")
    
    main(parser.parse_args())
