import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


sns.set_theme(
    context="paper",
    style="white"
)


def plot_fitness(*dataframes, num_of_iterations_to_plot=200, values_range=(0,None), plot_dimensions=(3.8,2.7), line_names=None, plot_title=None):
    disable_legend = False
    if line_names is None:
        disable_legend = True
        line_names = map(str, range(1, len(dataframes)+1))
        
    # Create a new DataFrame to hold all the data
    all_data = []
        
    for df, name in zip(dataframes, line_names):
        # Reset the index so "Iteration" becomes a column in the DataFrame and get the DataFrame in the long format
        df = df.reset_index()
        df = df.rename(columns={"index": "Iteration"})
        df = df.melt("Iteration", var_name="Task", value_name="Fitness")
        
        # Add a "Experiment name" column to the DataFrame for differenciation between the runs
        df["Experiment name"] = name
        
        # Append the data to the new DataFrame
        all_data.append(df)
        
    all_data = pd.concat(all_data)

    plt.figure(figsize=plot_dimensions)
    
    sns.lineplot(x="Iteration", y="Fitness", hue="Experiment name", errorbar="pi", data=all_data)
    
    if disable_legend:
        plt.legend().remove()
    else:
        plt.legend(loc="upper left", shadow=True)
    
    plt.ylim(values_range)
    plt.xlim(1, num_of_iterations_to_plot)
    if plot_title is not None:
        plt.title(plot_title)
    plt.tight_layout()
    plt.show()


def plot_evaluation_fitness(*dataframes, num_of_iterations_to_plot=200, values_range=(0,None), plot_dimensions=(3.8,2.7), line_names=None, plot_title=None):
    disable_legend = False
    if line_names is None:
        disable_legend = True
        line_names = map(str, range(1, len(dataframes)+1))
        
    # Create a new DataFrame to hold all the data
    all_data = []
        
    for df, name in zip(dataframes, line_names):
        # Reset the index so "Iteration" becomes a column in the DataFrame and get the DataFrame in the long format
        df = df.reset_index()
        df = df.rename(columns={"index": "Iteration"})
        df = df.melt("Iteration", var_name="Data type", value_name="Fitness")
        
        # Add a "Experiment name" column to the DataFrame for differenciation between the runs
        df["Experiment name"] = name
        
        # Append the data to the new DataFrame
        all_data.append(df)
        
    all_data = pd.concat(all_data)

    plt.figure(figsize=plot_dimensions)
    
    sns.lineplot(x="Iteration", y="Fitness", hue="Experiment name", style="Data type", style_order=["Best yet result", "Evaluation result"], data=all_data)
    
    if disable_legend:
        # Show only different line styles (data types) explanations, not color (experiment name)
        import matplotlib.lines as mlines
        
        plt.legend(handles=(
                mlines.Line2D([0], [0], color='black', linestyle="--", label="Evaluation result"),
                mlines.Line2D([0], [0], color='black', linestyle="-", label="Best yet result")
            ), loc="upper left", shadow=True)
        
    else:
        plt.legend(loc="upper left", shadow=True)
    
    plt.ylim(values_range)
    plt.xlim(1, num_of_iterations_to_plot)
    if plot_title is not None:
        plt.title(plot_title)
    plt.tight_layout()
    plt.show()


def plot_runtime(*dataframes, num_of_iterations_to_plot=200, values_range=(0,None), plot_dimensions=(3.8,2.7), line_names=None, plot_title=None):
    disable_legend = False
    if line_names is None:
        disable_legend = True
        line_names = map(str, range(1, len(dataframes)+1))
        
    # Create a new DataFrame to hold all the data
    all_data = []
        
    for df, name in zip(dataframes, line_names):
        # Reset the index so "Iteration" becomes a column in the DataFrame and get the DataFrame in the long format
        df = df.reset_index()
        df = df.rename(columns={"index": "Iteration"})
        df = df.melt("Iteration", var_name="Task", value_name="Runtime (timesteps)")
        
        # Add a "Experiment name" column to the DataFrame for differenciation between the runs
        df["Experiment name"] = name
        
        # Append the data to the new DataFrame
        all_data.append(df)
        
    all_data = pd.concat(all_data)

    plt.figure(figsize=plot_dimensions)
    
    sns.lineplot(x="Iteration", y="Runtime (timesteps)", hue="Experiment name", errorbar="pi", data=all_data)
    
    if disable_legend:
        plt.legend().remove()
    else:
        plt.legend(loc="upper left", shadow=True)
    
    plt.ylim(values_range)
    plt.xlim(1, num_of_iterations_to_plot)
    if plot_title is not None:
        plt.title(plot_title)
    plt.tight_layout()
    plt.show()


def plot_time(*dataframes, num_of_iterations_to_plot=200, values_range=(0,None), plot_dimensions=(3.8,2.7), line_names=None, plot_title=None):
    disable_legend = False
    if line_names is None:
        disable_legend = True
        line_names = map(str, range(1, len(dataframes)+1))
        
    # Create a new DataFrame to hold all the data
    all_data = []
        
    for df, name in zip(dataframes, line_names):
        # Reset the index so "Iteration" becomes a column in the DataFrame and get the DataFrame in the long format
        df = df.reset_index()
        df = df.rename(columns={"index": "Iteration", "Wall-clock time per iteration": "Wall-clock time per iteration (seconds)"})
        
        # Add a "Experiment name" column to the DataFrame for differenciation between the runs
        df["Experiment name"] = name
        
        # Append the data to the new DataFrame
        all_data.append(df)
        
    all_data = pd.concat(all_data)

    plt.figure(figsize=plot_dimensions)
    
    sns.lineplot(x="Iteration", y="Wall-clock time per iteration (seconds)", hue="Experiment name", data=all_data)
    
    if disable_legend:
        plt.legend().remove()
    else:
        plt.legend(loc="upper left", shadow=True)
    
    plt.ylim(values_range)
    plt.xlim(1, num_of_iterations_to_plot)
    if plot_title is not None:
        plt.title(plot_title)
    plt.tight_layout()
    plt.show()
