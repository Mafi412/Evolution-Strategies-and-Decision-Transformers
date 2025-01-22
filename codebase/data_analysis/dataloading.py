import pandas as pd
import os


def load_fitnesses_from_csv(path, nrows=200):
    data = pd.read_csv(
        path,
        sep=";",
        header=None,
        index_col=False,
        nrows=nrows
    )
    
    column_names = (str(i//2) + ("+" if i % 2 == 0 else "-") for i in range(data.shape[1]))
    data.columns = column_names
    
    # Adjust the index to start at 1 instead of 0 - row index = number of iteration
    data.index = data.index + 1
    
    return data


def load_evaluation_fitnesses_from_csv(path, nrows=200):
    data = pd.read_csv(
        path,
        sep="\t",
        header=0,
        index_col=False,
        nrows=nrows
    )
    
    # Adjust the index to start at 1 instead of 0 - row index = number of iteration
    data.index = data.index + 1
    
    return data


def load_runtimes_from_csv(path, nrows=200):
    data = pd.read_csv(
        path,
        sep=";",
        header=None,
        index_col=False,
        nrows=nrows
    )
    
    column_names = (str(i//2) + ("+" if i % 2 == 0 else "-") for i in range(data.shape[1]))
    data.columns = column_names
    
    # Adjust the index to start at 1 instead of 0 - row index = number of iteration
    data.index = data.index + 1
    
    return data


def load_times_from_csv(path, nrows=200):
    data = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=("Wall-clock time per iteration",),
        index_col=False,
        nrows=nrows
    )
    
    # Adjust the index to start at 1 instead of 0 - row index = number of iteration
    data.index = data.index + 1
    
    return data


def load_es_data(path, nrows=200):
    evaluation_fitness_path = path + ".evaluations.csv"
    fitness_path = path + ".fitness.csv"
    runtime_path = path + ".runtime.csv"
    time_path = path + ".time.csv"
    
    assert os.path.isfile(evaluation_fitness_path)
    assert os.path.isfile(fitness_path)
    assert os.path.isfile(runtime_path)
    assert os.path.isfile(time_path)
    
    evaluation_fitnesses = load_evaluation_fitnesses_from_csv(evaluation_fitness_path, nrows)
    fitnesses = load_fitnesses_from_csv(fitness_path, nrows)
    runtimes = load_runtimes_from_csv(runtime_path, nrows)
    iteration_times = load_times_from_csv(time_path, nrows)
    
    return evaluation_fitnesses, fitnesses, runtimes, iteration_times
