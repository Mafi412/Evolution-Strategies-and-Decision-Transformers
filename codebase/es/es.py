# Main logic of the distributed es utilizing MPI.

import mpi4py

mpi4py.rc.thread_level = "serialized"

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

import random
import time
import os
import sys

# Forbid multithreading for Numpy.
N_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS

import numpy as np
import torch

# Forbid multithreading for PyTorch.
torch.set_num_threads(1)

from . import funcs
from . import process_memory as pm
from es_utilities import utils

from tqdm import tqdm


def es(
    model,
    test_environment,
    size_of_population,
    num_of_iterations,
    main_seed,
    noise_deviation,
    weight_decay_factor,
    batch_size,
    update_vbn_stats_probability,
    path_for_checkpoints,
    logging_path
):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    
    # We change the exception hook so that in case we come across any exception in any process, all the processes shut down
    def mpiabort_excepthook(type, value, traceback):
        from traceback import print_exception
        print(flush=True)
        print_exception(None, value, traceback)
        comm.Abort(23)
    
    sys.excepthook = mpiabort_excepthook
    
    
    if size == 1:
        raise AssertionError("Only master is running! We need a master process and at least one worker process.")
    
    
    if rank == 0:
        print("Comm_world size:", size, flush=True)
        progress_bar = tqdm(total=num_of_iterations, file=sys.stdout)
        progress_bar.set_description("Setting everything up")
        
    
    # If main_seed is None we set it randomly, so we have at least somehow corresponding evaluations in the workers
    if rank == 0 and main_seed is None:
        main_seed = random.randint(0, 1e6)
        
    main_seed = comm.bcast(main_seed)
    pm.model = comm.bcast(model)
    pm.test_environment = comm.bcast(test_environment)
    
    
    # Set the seed
    utils.set_seed(main_seed)
    
    
    if rank > 0:
        # Those are not needed, nor wanted from the workers (and e.g. gym does sometimes output them)
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    
    # Control variables initialization
    
    ## Shared noise table
    if rank == 0:
        pm.shared_noise_table = utils.SharedNoiseTable(noise_deviation, main_seed)
        
    pm.shared_noise_table = comm.bcast(pm.shared_noise_table)
    
    ## Array of seeds of the individual tasks
    pm.seed_array = np.empty(size_of_population, dtype="i")
    
    ## Initial upper bound on the length of runtime for the evaluations
    pm.max_runtime = pm.test_environment.timestep_limit if pm.test_environment.timestep_limit is not None else 2147483647
    
    ## Array of ranks / weights for model update in each iteration (results from achieved fitnesses of individual noises)
    pm.rank_weights = np.empty(size_of_population, dtype="f")
    
    ## VBN-stats-related variables
    pm.update_vbn_stats_probability = update_vbn_stats_probability
    pm.sum_of_encountered_states = np.empty(pm.test_environment.state_shape, dtype="f")
    pm.sum_of_squares_of_encountered_states = np.empty(pm.test_environment.state_shape, dtype="f")
    pm.count_of_encountered_states = 0
                
                
    # Local variables
    if rank == 0:
        # Arrays for the results of the individual noise evaluations
        fitness_of_plus_noises = np.empty(size_of_population, dtype="f")
        fitness_of_minus_noises = np.empty(size_of_population, dtype="f")
        runtime_last_iteration_of_plus_noises = np.empty(size_of_population, dtype="i")
        runtime_last_iteration_of_minus_noises = np.empty(size_of_population, dtype="i")
        
    else:
        # Model to be used for the individual noise evaluations in the workers
        pm.test_model = pm.model.clone()
    
    
    # Prepare for logging in the master
    if rank == 0:
        # Creating non-existent directories on both logging and checkpoint paths and other path processing
        logging_path = os.path.normpath(logging_path)
        path_for_checkpoints = os.path.normpath(path_for_checkpoints)

        os.makedirs(os.path.dirname(logging_path), exist_ok=True)
        os.makedirs(os.path.dirname(path_for_checkpoints), exist_ok=True)
        
        # Logging-related local variables
        last_evaluation_result, last_evaluation_runtime, best_yet_iteration, best_return_yet, corresponding_runtime = float("nan"), float("nan"), None, -float("inf"), float("nan")
        evaluation_path = logging_path + ".evaluations"
        evaluation_csv_path = logging_path + ".evaluations.csv"
        fitness_path = logging_path + ".fitness.csv"
        runtime_path = logging_path + ".runtime.csv"
        time_path = logging_path + ".time.csv"
        
        # Clearing and preparing logfiles
        if os.path.exists(fitness_path):
            os.remove(fitness_path)

        if os.path.exists(runtime_path):
            os.remove(runtime_path)

        if os.path.exists(time_path):
            os.remove(time_path)

        if os.path.exists(evaluation_path):
            os.remove(evaluation_path)

        if os.path.exists(evaluation_csv_path):
            os.remove(evaluation_csv_path)

        with open(evaluation_csv_path, "a") as log:
                log.write(f"Evaluation result\tBest yet result\n")
    
    
    # Synchronization barrier after the setup phase and before the individual iterations
    comm.Barrier()
    

    # Iterations / Run of the program
    for iteration in range(num_of_iterations):
        
        # --- Iteration setup phase ---
        
        if rank == 0:
            # Report progress in the master
            progress_bar.set_description(f"Running iteration {iteration+1} | Best yet evaluation result (mean runtime) " + \
                f"being obtained after iteration {best_yet_iteration} - {best_return_yet:.4f} ({corresponding_runtime})")
        
            
            # Set beginning of the iteration for logging purposes
            iteration_start_time = time.time()
        
        # Set seed for this iteration in the master
        if rank == 0:
            utils.set_seed(main_seed + iteration)
        
        # Prepare seeds for individual tasks in the master process, then share them
        if rank == 0:
            for i in range(size_of_population):
                pm.seed_array[i] = random.randint(0, 2147483647)
                
        comm.Bcast(pm.seed_array)
    
        if rank == 0:
            # Reseting VBN-stats-related memory
            pm.sum_of_encountered_states = np.zeros(pm.test_environment.state_shape, dtype="f")
            pm.sum_of_squares_of_encountered_states = np.zeros(pm.test_environment.state_shape, dtype="f")
            pm.count_of_encountered_states = 0
        
        
        # --- Noises evaluation phase ---
        
        # Distribute the tasks and while they are being computed on the workers, perform evaluation of the model from the previous iteration.
        with MPICommExecutor(comm) as executor:
            if executor is not None: # In other words "execute just in root"
                noise_evaluation_results = executor.map(
                    funcs.noise_evaluations,
                    range(size_of_population),
                    (main_seed + iteration for _ in range(size_of_population))
                )
                
                
                # Evaluate the model resulting from the previous iteration and save it, if it is better than the best yet encountered
                last_evaluation_result, last_evaluation_runtime = funcs.evaluate_and_possibly_save(
                    pm.model,
                    pm.test_environment,
                    best_return_yet,
                    10,
                    path_for_checkpoints
                )
                
                if last_evaluation_result >= best_return_yet:
                    best_return_yet = last_evaluation_result
                    corresponding_runtime = last_evaluation_runtime
                    best_yet_iteration = iteration

                    progress_bar.set_description(f"Running iteration {iteration+1} | Best yet evaluation result (mean runtime) " + \
                        f"being obtained after iteration {best_yet_iteration} - {best_return_yet:.4f} ({corresponding_runtime})")
                
                # Log the evaluation results
                with open(evaluation_path, "a") as log:
                    log.write(f"Iteration {iteration} - Evaluation result (mean runtime): {last_evaluation_result} ({last_evaluation_runtime}) | Best yet: {best_return_yet} ({corresponding_runtime})\n")
                
                with open(evaluation_csv_path, "a") as log:
                    log.write(f"{last_evaluation_result}\t{best_return_yet}\n")
                
                
                # Process results of the individual noise evaluations
                for noise_evaluation_result in noise_evaluation_results:
                    if noise_evaluation_result.sum_ is not None:
                        pm.sum_of_encountered_states += np.reshape(noise_evaluation_result.sum_, pm.test_environment.state_shape)
                        pm.sum_of_squares_of_encountered_states += np.reshape(noise_evaluation_result.sum_of_squares, pm.test_environment.state_shape)
                        pm.count_of_encountered_states += noise_evaluation_result.count
                        
                    fitness_of_plus_noises[noise_evaluation_result.task_index] = noise_evaluation_result.fitness_of_plus_noise
                    fitness_of_minus_noises[noise_evaluation_result.task_index] = noise_evaluation_result.fitness_of_minus_noise
                    runtime_last_iteration_of_plus_noises[noise_evaluation_result.task_index] = noise_evaluation_result.runtime_of_plus_noise
                    runtime_last_iteration_of_minus_noises[noise_evaluation_result.task_index] = noise_evaluation_result.runtime_of_minus_noise
            
            
        if rank == 0:
            # Logging - fitness + runtime
            ## Fitness
            funcs.log_iteration_population_data(fitness_path, fitness_of_plus_noises, fitness_of_minus_noises)

            ## Runtime
            funcs.log_iteration_population_data(runtime_path, runtime_last_iteration_of_plus_noises, runtime_last_iteration_of_minus_noises)

            # Change fitness to the score / weight for update.
            modified_fitnesses = fitness_of_plus_noises - fitness_of_minus_noises
            modified_fitnesses[modified_fitnesses.argsort()] = np.arange(size_of_population) # From interval [0, size_of_population-1]
            modified_fitnesses /= (size_of_population - 1) # From interval [0,1]
            modified_fitnesses *= 2 # From interval [0,2]
            modified_fitnesses -= 1 # From interval [-1,1]
            pm.rank_weights[:] = modified_fitnesses
            
        # Share scores / weights for the individual noises and the VBN-stats update values
        comm.Bcast(pm.rank_weights)
        comm.Bcast(pm.sum_of_encountered_states)
        comm.Bcast(pm.sum_of_squares_of_encountered_states)
        pm.count_of_encountered_states = comm.bcast(pm.count_of_encountered_states)
    
    
        # Synchronization barrier after the evaluation phase and before the model update phase
        comm.Barrier()
        
        
        # --- Model update phase ---
        
        # Update the model
        funcs.update(
            weight_decay_factor,
            noise_deviation,
            batch_size,
        )
            
        # Update max_runtime and share it
        if rank == 0:
            total_runtime_last_iteration = np.sum(runtime_last_iteration_of_plus_noises) + np.sum(runtime_last_iteration_of_minus_noises)
            pm.max_runtime = total_runtime_last_iteration // size_of_population # = twice the mean number of steps taken per episode
            
        pm.max_runtime = comm.bcast(pm.max_runtime)


        # Synchronization barrier after the model update phase marking the end of an iteration
        comm.Barrier()
            
            
        if rank == 0:
            # Logging - iteration wall-clock duration
            iteration_duration = time.time() - iteration_start_time
            with open(time_path, "a") as log:
                log.write(str(iteration_duration) + "\n")
            
            
            progress_bar.update(1)
                
        
    if rank == 0:
        progress_bar.set_description(f"Running the last evaluation | Best yet evaluation result (mean runtime)" + \
            f"being obtained after iteration {best_yet_iteration} - {best_return_yet:.4f} ({corresponding_runtime})")
        
        # Final evaluation and saving of the resulting model
        last_evaluation_result, last_evaluation_runtime = funcs.evaluate_and_possibly_save(
            pm.model,
            pm.test_environment,
            best_return_yet,
            10,
            path_for_checkpoints
        )

        if last_evaluation_result >= best_return_yet:
            best_return_yet = last_evaluation_result
            corresponding_runtime = last_evaluation_runtime
            best_yet_iteration = num_of_iterations

        pm.model.save_parameters(path_for_checkpoints, "final_model")

        progress_bar.set_description(f"Finished | Best evaluation result (mean runtime): " + \
            f"{best_return_yet} ({corresponding_runtime}) from iteration {best_yet_iteration}")

        return pm.model
