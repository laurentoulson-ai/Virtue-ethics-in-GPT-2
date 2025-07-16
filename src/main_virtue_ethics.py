"""
Does what main.py does, but more specifically focuses on virtue ethics and neutral statements. Uses expanded data set of virtue ethics statements,
and excludes neurons with similar activation patterns to neutral statements, to better highlight specialized neurons for virtue ethics.
"""
from transformers import GPT2Tokenizer, GPT2Model
import matplotlib.pyplot as plt
import torch
import numpy as np
from collections import defaultdict
import csv

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
    return outputs.hidden_states

def load_statements():
    """Load virtue ethics and neutral statements"""
    virtue_ethics = []
    with open('data/virtue_ethics_statements.txt', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith("Statement"):
                virtue_ethics.append(line.strip())
    
    neutral = []
    with open('data/neutral_statements.txt', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith("Statement"):
                neutral.append(line.strip())
    
    return virtue_ethics, neutral

def calculate_specialization(virtue_acts, neutral_acts):
    """Very relaxed specialization detection with diagnostics"""
    specialized_neurons = []
    virtue_total = max(1, len(virtue_acts))
    neutral_total = max(1, len(neutral_acts))
    
    all_neurons = set(virtue_acts + neutral_acts)
    print(f"\nTotal unique neurons activated: {len(all_neurons)}")
    
    diagnostic_data = []
    
    for neuron in all_neurons:
        virtue_count = virtue_acts.count(neuron)
        neutral_count = neutral_acts.count(neuron)
        
        virtue_rate = virtue_count / virtue_total
        neutral_rate = neutral_count / neutral_total
        
        # Very relaxed criteria:
        # 1. Minimum activation in virtue ethics (3%)
        # 2. At least 1.3x higher than neutral
        # 3. No minimum absolute difference
        meets_criteria = (
            virtue_rate > 0.03 and 
            virtue_rate > 1.3 * neutral_rate
        )
        
        diagnostic_data.append((neuron, virtue_rate, neutral_rate, meets_criteria))
        
        if meets_criteria:
            specialized_neurons.append((neuron, virtue_rate, neutral_rate))
    
    # Print top candidates regardless of criteria
    print("\nTop 20 candidate neurons (sorted by virtue-neutral difference):")
    diagnostic_data.sort(key=lambda x: -(x[1] - x[2]))
    for neuron, v_rate, n_rate, meets in diagnostic_data[:20]:
        status = "★" if meets else " "
        print(f"{status} N{neuron:4}: Virtue {v_rate:.1%} vs Neutral {n_rate:.1%} | Ratio: {v_rate/max(0.001,n_rate):.1f}x")
    
    return sorted(specialized_neurons, key=lambda x: -(x[1] - x[2]))

def visualize_specialized_activations(specialized_neurons):
    """Visualize with more informative empty state"""
    if not specialized_neurons:
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, 
               "No specialized neurons found with current criteria\n" +
               "Check console output for diagnostic information",
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('virtue_ethics_specialization.png', dpi=300)
        plt.show()
        return
    
    neurons, virtue_rates, neutral_rates = zip(*specialized_neurons)
    
    x = np.arange(len(neurons))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    virtue_bars = ax.bar(x - width/2, virtue_rates, width, 
                        color='#59a14f', label='Virtue Ethics')
    neutral_bars = ax.bar(x + width/2, neutral_rates, width,
                         color='#4e79a7', label='Neutral')
    
    ax.set_title(f'Neuron Specialization for Virtue Ethics (n={len(neurons)} neurons)', fontsize=12)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Activation Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([f"N{int(n)}" for n in neurons], rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('virtue_ethics_specialization.png', dpi=300)
    plt.show()

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()
    
    virtue_ethics, neutral = load_statements()
    print(f"Loaded {len(virtue_ethics)} virtue ethics and {len(neutral)} neutral statements")
    
    # Extract activations with broader capture
    virtue_activations = []
    neutral_activations = []
    
    print("\nProcessing virtue ethics statements...")
    for text in virtue_ethics:
        hidden_states = get_activations(text, tokenizer, model)
        acts = hidden_states[6].mean(dim=1).squeeze()  # Middle layer
        top_neurons = np.argsort(np.abs(acts))[-20:]  # Top 20 neurons
        virtue_activations.extend(top_neurons.tolist())
    
    print("Processing neutral statements...")
    for text in neutral:
        hidden_states = get_activations(text, tokenizer, model)
        acts = hidden_states[6].mean(dim=1).squeeze()
        top_neurons = np.argsort(np.abs(acts))[-20:]  # Top 20 neurons
        neutral_activations.extend(top_neurons.tolist())
    
    print("\nCalculating specialization...")
    specialized_neurons = calculate_specialization(virtue_activations, neutral_activations)
    print(f"\nFound {len(specialized_neurons)} specialized neurons")
    
    # Visualize
    visualize_specialized_activations(specialized_neurons)

if __name__ == "__main__":
    main()