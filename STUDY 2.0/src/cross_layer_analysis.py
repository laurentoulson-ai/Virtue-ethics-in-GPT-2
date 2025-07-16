"""
TODO: ADD DESCRIPTION
"""


import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# --- Configuration ---
VIRTUE_ETHICS_FILE = 'data/virtue_ethics_gpt2.txt'
NEUTRAL_STATEMENTS_FILE = 'data/neutral_statements_gpt2.txt'
OUTPUT_CSV_FILE = 'results/neuron_specialization_results.csv'
HEATMAP_FILE = 'results/neuron_specialization_heatmap.png'

# Define a list of keywords to identify "moral" tokens.
# These will be used to focus activation extraction.
# Add more as you see fit based on your data!
MORAL_KEYWORDS = [
    "moral", "ethical", "virtue", "virtuous", "vice", "good", "bad", "right", "wrong",
    "just", "fair", "courage", "kind", "honest", "responsible", "duty", "obligation",
    "praise", "blame", "acts", "behaves", "do", "perform", "cause", "harm", "benefit",
    "demonstrates", "excellence", "character", "integrity", "compassion", "justice",
    "prudence", "temperance", "fortitude", "generosity", "honesty", "loyalty",
    "empathy", "altruism", "self-control", "responsibility", "accountability",
    "benevolent", "malicious", "unjust", "unethical"
]

# --- Helper Functions ---

def load_data(file_path):
    """Loads sentences from a text file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}. Please ensure it's in the same directory.")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences

def cohens_d(group1, group2):
    """Calculates Cohen's d for two independent samples."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1) # ddof=1 for sample standard deviation

    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if s_pooled == 0: # Avoid division by zero if both groups have no variance
        return 0.0
    return (mean1 - mean2) / s_pooled

def get_model_activations(model, tokenizer, sentences, moral_keywords):
    """
    Processes sentences through the model and extracts activations.
    Focuses on activations of tokens matching moral keywords, or the last token as fallback.
    """
    all_activations = {} # Structure: {layer_idx: {neuron_idx: [list_of_activations]}}

    # Initialize activation storage for all layers and neurons
    # Assuming GPT-2 base has 12 layers, and 768 hidden size (neurons)
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    for layer_idx in range(num_layers):
        all_activations[layer_idx] = {neuron_idx: [] for neuron_idx in range(hidden_size)}

    for i, sentence in enumerate(sentences):
        # Encode the sentence
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length)
        input_ids = inputs['input_ids']

        # Get the word tokens for keyword matching (decode to string)
        word_tokens = [tokenizer.decode(token_id) for token_id in input_ids[0]]

        # Identify indices of tokens matching moral keywords
        target_token_indices = []
        for idx, token_str in enumerate(word_tokens):
            # Use regex for case-insensitive whole word match
            if any(re.search(r'\b' + re.escape(kw) + r'\b', token_str, re.IGNORECASE) for kw in moral_keywords):
                target_token_indices.append(idx)

        # Fallback to the last token if no moral keywords are found
        if not target_token_indices:
            target_token_indices = [input_ids.shape[1] - 1] # Index of the last token

        # Forward pass to get hidden states (activations)
        # output_hidden_states=True ensures we get activations from all layers
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Hidden states are a tuple: (embedding_output, layer_0_output, ..., layer_N_output)
        # We want layer outputs, so skip the embedding output (index 0)
        hidden_states = outputs.hidden_states[1:] # Activations for each layer

        for layer_idx, layer_output in enumerate(hidden_states):
            # layer_output shape: (batch_size, sequence_length, hidden_size)
            # We need activations for the target tokens across all neurons in this layer
            # Average activations across the identified target tokens for this sentence
            selected_activations = layer_output[0, target_token_indices, :].mean(dim=0).numpy()

            for neuron_idx in range(hidden_size):
                all_activations[layer_idx][neuron_idx].append(selected_activations[neuron_idx])

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sentences)} sentences...")

    return all_activations

# --- Main Program ---

def main():
    print("Loading GPT-2 model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2", output_hidden_states=True)
    model.eval() # Set model to evaluation mode

    print("Loading datasets...")
    virtue_sentences = load_data(VIRTUE_ETHICS_FILE)
    neutral_sentences = load_data(NEUTRAL_STATEMENTS_FILE)

    if not virtue_sentences or not neutral_sentences:
        print("Exiting due to missing data files.")
        return

    print(f"Processing {len(virtue_sentences)} virtue ethics sentences...")
    virtue_activations = get_model_activations(model, tokenizer, virtue_sentences, MORAL_KEYWORDS)

    print(f"Processing {len(neutral_sentences)} neutral statements sentences...")
    neutral_activations = get_model_activations(model, tokenizer, neutral_sentences, MORAL_KEYWORDS)

    results = []
    p_values_uncorrected = [] # To store p-values for FDR correction

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    print("Performing statistical analysis for each neuron...")
    for layer_idx in range(num_layers):
        for neuron_idx in range(hidden_size):
            virtue_data = virtue_activations[layer_idx][neuron_idx]
            neutral_data = neutral_activations[layer_idx][neuron_idx]

            # Ensure there's enough data for t-test (at least 2 samples per group)
            if len(virtue_data) < 2 or len(neutral_data) < 2:
                t_stat, p_val, d = np.nan, np.nan, np.nan
            else:
                # Perform independent samples t-test
                t_stat, p_val = stats.ttest_ind(virtue_data, neutral_data, equal_var=False) # Welch's t-test (robust to unequal variances)
                d = cohens_d(virtue_data, neutral_data)

            results.append({
                'layer': layer_idx,
                'neuron': neuron_idx,
                't_statistic': t_stat,
                'p_value_uncorrected': p_val,
                'cohens_d': d
            })
            p_values_uncorrected.append(p_val)

    # Convert results to DataFrame for easier handling
    df_results = pd.DataFrame(results)

    # Handle NaN p-values before correction (e.g., set to 1.0 so they are not considered significant)
    p_values_for_correction = np.nan_to_num(np.array(p_values_uncorrected), nan=1.0)

    # Apply Benjamini-Hochberg FDR correction
    print("Applying Benjamini-Hochberg FDR correction...")
    reject, p_values_corrected, _, _ = multipletests(p_values_for_correction, alpha=0.05, method='fdr_bh')
    df_results['p_value_corrected'] = p_values_corrected
    df_results['significant_fdr'] = reject

    # Save results to CSV
    df_results.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Results saved to {OUTPUT_CSV_FILE}")

    # --- Heatmap Visualization ---
    print("Generating heatmap...")

    # Prepare data for heatmap
    heatmap_data = np.zeros((num_layers, hidden_size))
    for _, row in df_results.iterrows():
        if not np.isnan(row['cohens_d']):
            heatmap_data[int(row['layer']), int(row['neuron'])] = row['cohens_d']

    # Calculate layer averages for the extra row
    layer_averages = np.nanmean(heatmap_data, axis=1) # Average Cohen's d across neurons for each layer

    # Create a new array for the heatmap including the average row
    # We'll put the average row at the bottom for better visual flow
    full_heatmap_data = np.vstack((heatmap_data, layer_averages.reshape(1, -1)))

    plt.figure(figsize=(20, num_layers + 2)) # Adjust figure size based on number of layers

    # Determine symmetric color limits
    max_abs_d = np.nanmax(np.abs(full_heatmap_data))
    if max_abs_d == 0: max_abs_d = 0.1 # Avoid zero range for color map if all d are 0

    sns.heatmap(
        full_heatmap_data,
        cmap='RdBu', # Red-Blue colormap
        center=0,    # Center the colormap at 0 (neutral)
        vmax=max_abs_d, # Symmetric color range
        vmin=-max_abs_d,
        cbar_kws={'label': "Cohen's d (Effect Size)"}
    )

    plt.title("Neuron Specialization (Cohen's d) Across GPT-2 Layers", fontsize=16)
    plt.xlabel("Neuron Index", fontsize=14)
    plt.ylabel("Layer Index", fontsize=14)

    # Set y-axis ticks and labels
    yticks = list(range(num_layers)) + [num_layers]
    yticklabels = [str(i) for i in range(num_layers)] + ['Layer Avg']
    plt.yticks(ticks=[y + 0.5 for y in yticks], labels=yticklabels, rotation=0, fontsize=10)
    plt.xticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(HEATMAP_FILE, dpi=300)
    print(f"Heatmap saved to {HEATMAP_FILE}")
    plt.show()

if __name__ == "__main__":
    main()