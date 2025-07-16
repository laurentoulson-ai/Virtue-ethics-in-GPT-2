"""
Builds from main_virtue_ethics.py which finds two neurons that activate much more strongly for virtue ethics statements than neutral ones
Now visualises each statement's activation for these neurons in a heatmap to help identify which statements the neurons are most sensitive to
"""
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Configuration
NEURONS_OF_INTEREST = [87, 745]  # Our specialized neurons
LAYER = 6                         # Layer showing best specialization
MAX_STATEMENT_LENGTH = 100        # Characters to display in table
SAVE_FULL_DATA = True             # Save complete data to CSV

def load_virtue_ethics_statements():
    """Load virtue ethics statements from file"""
    virtue_ethics = []
    with open('data/virtue_ethics_statements.txt', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith("Statement"):
                virtue_ethics.append(line.strip())
    return virtue_ethics

def get_neuron_activations(text, tokenizer, model):
    """Get absolute activation values for target neurons"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[LAYER].mean(dim=1).squeeze().abs()

def create_activation_dataframe():
    """Create dataframe with statements and activation values"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()
    
    statements = load_virtue_ethics_statements()
    data = []
    
    for text in statements:
        activations = get_neuron_activations(text, tokenizer, model)
        data.append({
            'Full_Statement': text,
            'Statement_Preview': text[:MAX_STATEMENT_LENGTH] + ('...' if len(text) > MAX_STATEMENT_LENGTH else ''),
            'N87': float(activations[87]),
            'N745': float(activations[745]),
            'Difference(N87-N745)': float(activations[87] - activations[745])
        })
    
    return pd.DataFrame(data)

def create_color_mapping(values):
    """Create red-white-blue color mapping normalized to data range"""
    max_val = max(values.max(), -values.min())
    return {
        'red': [(0, 0.0, 1.0), (0.5, 1.0, 1.0), (1, 1.0, 1.0)],
        'green': [(0, 0.0, 1.0), (0.5, 1.0, 1.0), (1, 0.0, 0.0)],
        'blue': [(0, 1.0, 1.0), (0.5, 1.0, 1.0), (1, 0.0, 0.0)]
    }

def plot_activation_table(df):
    """Create publication-quality table with color coding"""
    plt.figure(figsize=(14, max(6, len(df)/2)))
    ax = plt.gca()
    ax.axis('off')
    
    # Prepare data for display
    display_df = df[['Statement_Preview', 'N87', 'N745', 'Difference(N87-N745)']].copy()
    display_df.columns = ['Statement (Preview)', 'N87 Activation', 'N745 Activation', 'Difference (N87-N745)']
    
    # Create color mappers
    n87_colors = plt.cm.RdYlBu_r((df['N87'] - df['N87'].min()) / (df['N87'].max() - df['N87'].min()))
    n745_colors = plt.cm.RdYlBu_r((df['N745'] - df['N745'].min()) / (df['N745'].max() - df['N745'].min()))
    diff_colors = plt.cm.RdYlBu_r((df['Difference(N87-N745)'] + df['Difference(N87-N745)'].abs().max()) / 
                  (2 * df['Difference(N87-N745)'].abs().max()))
    
    # Create table
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc='center',
        cellColours=[['white'] + [n87_colors[i], n745_colors[i], diff_colors[i]] for i in range(len(df))],
        colColours=['#f7f7f7', '#ff9999', '#99ccff', '#dddddd'],
        cellLoc='center'
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    
    # Header styling
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4e79a7')
            cell.set_text_props(color='white', weight='bold')
        elif j == 0:  # Statement column
            cell.set_facecolor('#f0f0f0')
    
    plt.title('Virtue Ethics Neuron Activation Patterns\n(Red = Stronger Activation, Blue = Weaker Activation)', 
              pad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig('virtue_ethics_activation_table.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating activation data...")
    activation_df = create_activation_dataframe().sort_values('N87', ascending=False)
    
    if SAVE_FULL_DATA:
        activation_df.to_csv('virtue_ethics_activation_data.csv', index=False)
        print("Full data saved to 'virtue_ethics_activation_data.csv'")
    
    print("\nTop 3 Statements by N87 Activation:")
    print(activation_df[['Statement_Preview', 'N87', 'N745']].head(3))
    
    print("\nGenerating color-coded table...")
    plot_activation_table(activation_df)
    print("Table visualization saved to 'virtue_ethics_activation_table.png'")
