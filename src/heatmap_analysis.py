import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from importlib.util import spec_from_file_location, module_from_spec
import sys
from pathlib import Path

def import_main_module():
    # Dynamically import main.py to access its functions and variables
    main_path = Path(__file__).parent / "main.py"
    spec = spec_from_file_location("main", main_path)
    main = module_from_spec(spec)
    sys.modules["main"] = main
    spec.loader.exec_module(main)
    return main

def extract_activation_data(main_module):
    # Run the main function to populate type_neurons
    main_module.main()
    
    # Access the type_neurons dictionary from main.py's namespace
    if hasattr(main_module, 'get_type_neurons'):
        type_neurons = main_module.get_type_neurons()
    else:
        raise ValueError("Could not find type_neurons data in main.py")
    
    # Calculate activation rates per neuron per type
    neuron_activation_rates = defaultdict(dict)
    all_neurons = set()
    
    # First pass to get all unique neurons
    for stmt_type, neurons in type_neurons.items():
        all_neurons.update(neurons)
    
    # Second pass to calculate normalised activation rates
    for stmt_type, neurons in type_neurons.items():
        total_statements = max(1, len(neurons)/10)  # Since each statement adds 10 neurons
        for neuron in all_neurons:
            count = neurons.count(neuron)
            neuron_activation_rates[neuron][stmt_type] = count / total_statements
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(neuron_activation_rates, orient='index')
    df.index.name = 'Neuron_Index'
    
    # Reorder columns to match original visualisation
    column_order = ['neutral', 'Deontological', 'Virtue Ethics', 'Utilitarian', 
                   'Controversial/Contextual', 'Rights-based']
    df = df.reindex(columns=[col for col in column_order if col in df.columns])
    
    return df

def plot_heatmap(df):
    plt.figure(figsize=(12, 8))
    
    # Use a traditional heatmap colormap: red for high, blue for low, white for neutral
    cmap = 'seismic'
    
    sns.heatmap(
        df.T,  # Transpose to have statement types as rows
        cmap=cmap,
        vmin=0,  # Minimum value for the heatmap
        vmax=1,  # Maximum value for the heatmap
        cbar_kws={'label': 'Normalized Activation Rate'},  # Add a color bar label
        linewidths=0.5,  # Add gridlines between cells
        linecolor='white'  # Gridline color
    )
    
    # Add titles and labels
    plt.title('Neuron Activation Patterns by Statement Type', fontsize=14)
    plt.xlabel('Neuron Index', fontsize=12)
    plt.ylabel('Statement Type', fontsize=12)
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    
    # Adjust layout to prevent cutoff
    plt.tight_layout()
    plt.savefig('neuron_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Extracting activation data from main.py...")
    main_module = import_main_module()
    
    print("Processing activation patterns...")
    activation_df = extract_activation_data(main_module)
    
    print("Generating heatmap visualization...")
    plot_heatmap(activation_df)
    print("Heatmap saved to 'neuron_heatmap.png'")