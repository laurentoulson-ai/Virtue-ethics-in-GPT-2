"""
Based on main_virtue_ethics.py but inspects the N87 and N745 neurons specifically, which are known to activate strongly for virtue ethics statements.
Tests whether these neurons are only active for virtue ethics statements or if they are also active for other moral types - uses expanded data sets
for both deontological and utilitarian statements, as well as neutral ones.
"""
from transformers import GPT2Tokenizer, GPT2Model
import matplotlib.pyplot as plt
import torch
import numpy as np
import csv
from collections import defaultdict

# Configure matplotlib
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.4,
    'font.size': 10
})

def get_activations(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[6].mean(dim=1).squeeze().abs()  # Layer 6 activations

def load_statements():
    """Load all statement types"""
    statements = {}
    
    # Neutral statements
    with open('data/neutral_statements.txt', 'r') as f:
        statements['neutral'] = [line.strip() for line in f if line.strip() and not line.startswith("Statement")]
    
    # Virtue ethics (already processed but included for comparison)
    with open('data/virtue_ethics_statements.txt', 'r') as f:
        statements['virtue_ethics'] = [line.strip() for line in f if line.strip() and not line.startswith("Statement")]
    
    # Deontological statements
    with open('data/deontological_statements_300.txt', 'r') as f:
        statements['deontological'] = [line.strip() for line in f if line.strip()]
    
    # Utilitarian statements
    with open('data/utilitarian_statements_300.txt', 'r') as f:
        statements['utilitarian'] = [line.strip() for line in f if line.strip()]
    
    return statements

def calculate_activation_rates(statements, tokenizer, model, target_neurons):
    """Calculate activation rates for specific neurons across all statement types"""
    activation_counts = {stmt_type: defaultdict(int) for stmt_type in statements}
    
    for stmt_type, stmts in statements.items():
        for text in stmts:
            activations = get_activations(text, tokenizer, model)
            top_neurons = np.argsort(activations)[-20:]  # Top 20 activated neurons
            
            for neuron in target_neurons:
                if neuron in top_neurons:
                    activation_counts[stmt_type][neuron] += 1
    
    # Convert counts to rates
    activation_rates = {}
    for stmt_type in activation_counts:
        activation_rates[stmt_type] = {
            neuron: count / len(statements[stmt_type]) 
            for neuron, count in activation_counts[stmt_type].items()
        }
    
    return activation_rates

def plot_comparison(activation_rates, target_neurons):
    """Create bar plot comparing activation across moral types"""
    stmt_types = ['neutral', 'virtue_ethics', 'deontological', 'utilitarian']
    colors = ['#4e79a7', '#59a14f', '#e15759', '#edc948']  # Blue, Green, Red, Yellow
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Position of bars
    x = np.arange(len(target_neurons))
    width = 0.2
    
    for i, stmt_type in enumerate(stmt_types):
        rates = [activation_rates[stmt_type].get(neuron, 0) for neuron in target_neurons]
        ax.bar(x + i*width, rates, width, label=stmt_type.capitalize(), color=colors[i])
    
    # Add ratio annotations
    for j, neuron in enumerate(target_neurons):
        virtue_rate = activation_rates['virtue_ethics'].get(neuron, 0)
        deont_rate = activation_rates['deontological'].get(neuron, 0)
        util_rate = activation_rates['utilitarian'].get(neuron, 0)
        neutral_rate = activation_rates['neutral'].get(neuron, 0.001)  # Avoid division by zero
        
        # Calculate ratios
        virtue_ratio = virtue_rate / neutral_rate
        deont_ratio = deont_rate / neutral_rate
        util_ratio = util_rate / neutral_rate
        
        # Annotate above each neuron group
        ax.text(x[j] + 1.5*width, max(rates) + 0.05, 
               f'N{neuron} Ratios:\n'
               f'Virtue: {virtue_ratio:.1f}x\n'
               f'Deont: {deont_ratio:.1f}x\n'
               f'Util: {util_ratio:.1f}x',
               ha='center', fontsize=8)
    
    ax.set_title('Neuron Activation Comparison Across Moral Frameworks (Layer 6)', fontsize=12)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Activation Rate')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([f'N{neuron}' for neuron in target_neurons])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('moral_framework_comparison.png', dpi=300)
    plt.show()

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()
    
    # Target neurons from previous analysis
    target_neurons = [87, 745]
    
    # Load all statement types
    statements = load_statements()
    print(f"Loaded statements: { {k: len(v) for k, v in statements.items()} }")
    
    # Calculate activation rates
    activation_rates = calculate_activation_rates(statements, tokenizer, model, target_neurons)
    
    # Print raw rates
    print("\nActivation Rates:")
    for stmt_type in activation_rates:
        print(f"\n{stmt_type.upper()}:")
        for neuron in target_neurons:
            rate = activation_rates[stmt_type].get(neuron, 0)
            print(f"N{neuron}: {rate:.1%}")
    
    # Visualize comparison
    plot_comparison(activation_rates, target_neurons)

if __name__ == "__main__":
    main()