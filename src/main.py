from transformers import GPT2Tokenizer, GPT2Model
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import csv
import numpy as np
from collections import defaultdict

# === Matplotlib Configuration ===
# Configure matplotlib to use a clean style for plots
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',  # Set the background color of figures to white
    'axes.grid': True,           # Enable grid lines on plots
    'grid.linestyle': ':',       # Use dotted grid lines
    'grid.alpha': 0.4,           # Set grid line transparency
    'font.size': 10              # Set default font size
})

# === Function: get_activations ===
# This function extracts the hidden states (activations) from the GPT-2 model for a given text input
def get_activations(text, tokenizer, model):
    # Tokenize the input text and convert it to tensors
    inputs = tokenizer(text, return_tensors="pt")
    # Perform inference with the model without updating weights (no_grad)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Return the hidden states (activations) from the model
    return outputs.hidden_states

# === Function: load_statements_with_types ===
# This function loads moral and neutral statements from text files and categorizes them by type
def load_statements_with_types():
    """Load statements with moral type information"""
    moral = []  # List to store moral statements
    moral_types = []  # List to store corresponding moral types
    # Read moral statements and their types from a tab-delimited file
    with open('data/moral_statements.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            if row:  # Ensure the row is not empty
                moral.append(row[0].strip())  # Add the statement
                moral_types.append(row[1].strip())  # Add the moral type
    
    neutral = []  # List to store neutral statements
    # Read neutral statements from a file
    with open('data/neutral_statements.txt', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith("Statement"):  # Skip empty lines and headers
                neutral.append(line.strip())  # Add the statement
    
    # Return the lists of moral statements, neutral statements, and moral types
    return moral, neutral, moral_types

# === Function: visualize_activations ===
# This function creates a bar plot to visualize neuron activation patterns for different statement types
def visualize_activations(type_neurons):
    """Create normalized visualization of neuron activations"""
    # Calculate the number of statements for each type
    type_counts = {k: max(1, len(v)) for k, v in type_neurons.items()}
    
    # Prepare data for plotting: count activations per neuron per type
    neuron_data = defaultdict(lambda: defaultdict(int))
    all_neurons = set()  # Set to store all unique neurons
    
    # Count activations for each neuron and statement type
    for stmt_type, neurons in type_neurons.items():
        for neuron in neurons:
            neuron_data[neuron][stmt_type] += 1
            all_neurons.add(neuron)
    
    # Sort neurons by their total activation frequency
    sorted_neurons = sorted(all_neurons, key=lambda n: -sum(neuron_data[n].values()))
    
    # Define colors for each statement type
    colors = {
        'neutral': '#4e79a7',
        'Deontological': '#e15759',
        'Virtue Ethics': '#59a14f',
        'Utilitarian': '#edc948',
        'Controversial/Contextual': '#af7aa1',
        'Rights-based': '#ff9da7'
    }
    moral_types = sorted([k for k in type_neurons.keys() if k != 'neutral'])
    
    # Create the plot
    num_neurons = len(sorted_neurons)
    fig, ax = plt.subplots(figsize=(max(12, num_neurons*0.35), 8))
    
    # Plot neutral activations (normalized)
    x = np.arange(num_neurons)
    bar_width = 0.8
    neutral_norm = [neuron_data[n].get('neutral', 0)/type_counts['neutral'] for n in sorted_neurons]
    ax.bar(x, neutral_norm, width=bar_width, color=colors['neutral'], label=f'Neutral (n={type_counts["neutral"]})')
    
    # Plot moral types stacked (normalized)
    bottoms = np.array(neutral_norm)
    for m_type in moral_types:
        norm_vals = [neuron_data[n].get(m_type, 0)/type_counts[m_type] for n in sorted_neurons]
        bars = ax.bar(x, norm_vals, width=bar_width, bottom=bottoms, color=colors[m_type], label=f'{m_type} (n={type_counts[m_type]})')
        bottoms += np.array(norm_vals)
    
    # Customize the plot
    ax.set_title('Normalized Neuron Activation Patterns by Statement Type', pad=20, fontsize=12)
    ax.set_xlabel('Neuron Index', fontsize=11)
    ax.set_ylabel('Activation Rate (per statement)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N{int(n)}" for n in sorted_neurons], rotation=45, ha='right', fontsize=9)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    
    # Save and display the plot
    plt.tight_layout()
    plt.savefig('neuron_activations_normalized.png', dpi=300, bbox_inches='tight')
    plt.show()

# === Function: analyze_activation_patterns ===
# This function analyzes and compares neuron activation patterns across different moral types
def analyze_activation_patterns(type_neurons):
    """Compare activation patterns across moral types"""
    print("\n=== Moral Type Activation Comparison ===")
    
    # Calculate metrics for each moral type
    activation_metrics = {}
    for m_type, neurons in type_neurons.items():
        unique, counts = np.unique(neurons, return_counts=True)
        activation_metrics[m_type] = {
            'total_acts': len(neurons),
            'unique_neurons': len(unique),
            'avg_acts_per_neuron': np.mean(counts) if len(counts) > 0 else 0,
            'specialization': np.mean(counts)/len(neurons) if len(neurons) > 0 else 0
        }
    
    # Print a comparison table
    print("{:<25} {:<15} {:<15} {:<20} {:<15}".format(
        "Type", "Total Activations", "Unique Neurons", 
        "Avg Acts/Neuron", "Specialization"))
    for m_type, metrics in activation_metrics.items():
        print("{:<25} {:<15} {:<15} {:<20.2f} {:<15.2f}".format(
            m_type, metrics['total_acts'], metrics['unique_neurons'],
            metrics['avg_acts_per_neuron'], metrics['specialization']))
    
    # Identify neurons specialized for each moral type
    print("\n=== Specialized Neurons ===")
    moral_types = [k for k in type_neurons.keys() if k != 'neutral']
    all_moral_neurons = set().union(*[set(type_neurons[t]) for t in moral_types])
    
    for m_type in moral_types:
        other_types = [t for t in moral_types if t != m_type]
        other_neurons = set().union(*[set(type_neurons[t]) for t in other_types])
        specialized = set(type_neurons[m_type]) - other_neurons
        if specialized:
            print(f"{m_type}-specific neurons: {sorted(specialized)}")

# === Global Variable: type_neurons ===
# A dictionary to store neuron activations categorized by statement type
global type_neurons
type_neurons = defaultdict(list)

# === Main Function ===
# This function orchestrates the loading of data, extraction of activations, and analysis
def main():
    # Initialize the GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()

    # Load moral and neutral statements
    moral, neutral, moral_types = load_statements_with_types()
    print(f"Loaded {len(moral)} moral and {len(neutral)} neutral statements")

    # Extract neuron activations for each statement
    global type_neurons
    type_neurons = defaultdict(list)
    for i, text in enumerate(moral):
        hidden_states = get_activations(text, tokenizer, model)
        activations = hidden_states[6].mean(dim=1).squeeze()
        top_neurons = np.argsort(np.abs(activations))[-10:]  # Top 10 neurons
        type_neurons[moral_types[i]].extend(top_neurons.tolist())
    
    for text in neutral:
        hidden_states = get_activations(text, tokenizer, model)
        activations = hidden_states[6].mean(dim=1).squeeze()
        top_neurons = np.argsort(np.abs(activations))[-10:]  # Top 10 neurons
        type_neurons['neutral'].extend(top_neurons.tolist())
    
    # Generate visualizations and analyze patterns
    visualize_activations(type_neurons)
    analyze_activation_patterns(type_neurons)
    print(f"type_neurons populated with {len(type_neurons)} types")

# === Function: get_type_neurons ===
# This function returns the global type_neurons dictionary
def get_type_neurons():
    return type_neurons


if __name__ == "__main__":
    main()