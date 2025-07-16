"""
After filtering my virtue ethics statements by the top 20% of statements that fire for N87, this programme now attempts to identify if there is a specialised
circuit for 'prosocial actions' - which N87 seems to be specialised for. Are other layers or neurons even more specialised?
This programme searches all 12 layers of GPT-2 to find where there is highest activation for the top 20% of virtue ethics statements compared to neutral statements.
"""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import os

class VirtueEthicsAnalyzer:
    def __init__(self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the analyzer with GPT-2 model and tokenizer
        """
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store activations
        self.activations = {}
        self.hooks = []
        
    def load_datasets(self, virtue_path, neutral_path):
        """
        Load virtue ethics and neutral statement datasets
        """
        with open(virtue_path, 'r', encoding='utf-8') as f:
            virtue_statements = [line.strip() for line in f if line.strip()]
        
        with open(neutral_path, 'r', encoding='utf-8') as f:
            neutral_statements = [line.strip() for line in f if line.strip()]
            
        print(f"Loaded {len(virtue_statements)} virtue ethics statements")
        print(f"Loaded {len(neutral_statements)} neutral statements")
        
        return virtue_statements, neutral_statements
    
    def register_hooks(self, target_layers=None):
        """
        Register forward hooks to capture activations from specified layers
        """
        if target_layers is None:
            target_layers = list(range(12))  # All layers for GPT-2
        
        def get_activation(name):
            def hook(model, input, output):
                # For transformer blocks, we want the MLP output
                if hasattr(output, 'last_hidden_state'):
                    self.activations[name] = output.last_hidden_state.detach()
                elif isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        # Clear previous hooks
        self.clear_hooks()
        
        # Register hooks for each target layer
        for layer_idx in target_layers:
            layer_name = f"layer_{layer_idx}"
            if layer_idx < len(self.model.transformer.h):
                hook = self.model.transformer.h[layer_idx].mlp.register_forward_hook(
                    get_activation(layer_name)
                )
                self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self, texts, batch_size=8):
        """
        Get neural activations for a list of texts
        """
        all_layer_activations = defaultdict(list)
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Clear previous activations
            self.activations = {}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Store activations for each layer
            for layer_name, activation in self.activations.items():
                # Average across sequence length and batch
                avg_activation = activation.mean(dim=1)  # Average over sequence length
                all_layer_activations[layer_name].append(avg_activation.cpu())
        
        # Concatenate all batches
        layer_activations = {}
        for layer_name, activations_list in all_layer_activations.items():
            layer_activations[layer_name] = torch.cat(activations_list, dim=0)
        
        return layer_activations
    
    def analyze_specialization(self, virtue_statements, neutral_statements, 
                             target_layers=None, top_k=10, min_activation_threshold=0.01):
        """
        Analyze neural specialization across layers with improved methodology
        """
        if target_layers is None:
            target_layers = list(range(12))
        
        # Register hooks for target layers
        self.register_hooks(target_layers)
        
        print("Getting activations for virtue ethics statements...")
        virtue_activations = self.get_activations(virtue_statements)
        
        print("Getting activations for neutral statements...")
        neutral_activations = self.get_activations(neutral_statements)
        
        # Analyze each layer
        results = {}
        
        for layer_idx in target_layers:
            layer_name = f"layer_{layer_idx}"
            
            if layer_name not in virtue_activations:
                continue
                
            virtue_acts = virtue_activations[layer_name]
            neutral_acts = neutral_activations[layer_name]
            
            # Calculate mean and std for each neuron
            virtue_mean = virtue_acts.mean(dim=0)
            neutral_mean = neutral_acts.mean(dim=0)
            virtue_std = virtue_acts.std(dim=0)
            neutral_std = neutral_acts.std(dim=0)
            
            # Calculate absolute difference (more robust than ratio)
            abs_difference = torch.abs(virtue_mean - neutral_mean)
            
            # Filter out neurons with very low activations in both conditions
            both_means = torch.abs(virtue_mean) + torch.abs(neutral_mean)
            active_mask = both_means > min_activation_threshold
            
            # Calculate multiple metrics
            # 1. Ratio method (with better handling of small values)
            ratio_denominator = torch.maximum(torch.abs(neutral_mean), torch.tensor(0.001))
            specialization_ratio = torch.abs(virtue_mean) / ratio_denominator
            
            # 2. Effect size (Cohen's d)
            pooled_std = torch.sqrt((virtue_std**2 + neutral_std**2) / 2)
            effect_size = torch.abs(virtue_mean - neutral_mean) / (pooled_std + 1e-8)
            
            # 3. Statistical significance proxy (t-statistic)
            se_diff = torch.sqrt(virtue_std**2/len(virtue_statements) + neutral_std**2/len(neutral_statements))
            t_statistic = torch.abs(virtue_mean - neutral_mean) / (se_diff + 1e-8)
            
            # Apply active mask and get top neurons by different metrics
            if active_mask.sum() > 0:
                # Use effect size as primary metric (more robust)
                filtered_effect_size = effect_size * active_mask.float()
                top_indices = torch.argsort(filtered_effect_size, descending=True)[:top_k]
            else:
                # Fallback to ratio method
                top_indices = torch.argsort(specialization_ratio, descending=True)[:top_k]
            
            layer_results = []
            for idx in top_indices:
                neuron_idx = idx.item()
                
                # Skip if not active
                if not active_mask[neuron_idx]:
                    continue
                
                ratio = specialization_ratio[neuron_idx].item()
                virtue_val = virtue_mean[neuron_idx].item()
                neutral_val = neutral_mean[neuron_idx].item()
                effect_sz = effect_size[neuron_idx].item()
                t_stat = t_statistic[neuron_idx].item()
                
                layer_results.append({
                    'neuron': neuron_idx,
                    'specialization_ratio': ratio,
                    'virtue_activation': virtue_val,
                    'neutral_activation': neutral_val,
                    'difference': virtue_val - neutral_val,
                    'effect_size': effect_sz,
                    't_statistic': t_stat,
                    'virtue_std': virtue_std[neuron_idx].item(),
                    'neutral_std': neutral_std[neuron_idx].item()
                })
            
            results[layer_idx] = layer_results
            
            print(f"\nLayer {layer_idx} - Top {top_k} specialized neurons (by effect size):")
            for i, result in enumerate(layer_results):
                print(f"  {i+1}. Neuron {result['neuron']:3d}: "
                      f"Effect size: {result['effect_size']:.2f}, "
                      f"Ratio: {result['specialization_ratio']:.2f}x, "
                      f"t-stat: {result['t_statistic']:.2f}")
                print(f"      Virtue: {result['virtue_activation']:.4f}±{result['virtue_std']:.4f}, "
                      f"Neutral: {result['neutral_activation']:.4f}±{result['neutral_std']:.4f}")
        
        return results
    
    def visualize_specialization(self, results, save_path="virtue_specialization.png"):
        """
        Create visualizations of neural specialization across layers
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Heatmap of top specialization ratios by layer
        layers = sorted(results.keys())
        top_ratios = []
        
        for layer in layers:
            layer_ratios = [r['specialization_ratio'] for r in results[layer][:5]]
            top_ratios.append(layer_ratios)
        
        ax1 = axes[0, 0]
        sns.heatmap(np.array(top_ratios).T, 
                   xticklabels=[f"Layer {l}" for l in layers],
                   yticklabels=[f"Top {i+1}" for i in range(5)],
                   annot=True, fmt='.1f', cmap='viridis', ax=ax1)
        ax1.set_title('Top 5 Specialization Ratios by Layer')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Neuron Rank')
        
        # 2. Max specialization ratio per layer
        ax2 = axes[0, 1]
        max_ratios = [max(r['specialization_ratio'] for r in results[layer]) 
                     for layer in layers]
        ax2.bar(layers, max_ratios, color='skyblue', alpha=0.7)
        ax2.set_title('Maximum Specialization Ratio by Layer')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Max Specialization Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Distribution of specialization ratios
        ax3 = axes[1, 0]
        all_ratios = []
        layer_labels = []
        
        for layer in layers:
            ratios = [r['specialization_ratio'] for r in results[layer]]
            all_ratios.extend(ratios)
            layer_labels.extend([f"Layer {layer}"] * len(ratios))
        
        df = pd.DataFrame({'Layer': layer_labels, 'Specialization_Ratio': all_ratios})
        sns.boxplot(data=df, x='Layer', y='Specialization_Ratio', ax=ax3)
        ax3.set_title('Distribution of Specialization Ratios')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Highlight Layer 6 Neuron 87 if present
        ax4 = axes[1, 1]
        if 6 in results:
            layer_6_results = results[6]
            neurons = [r['neuron'] for r in layer_6_results]
            ratios = [r['specialization_ratio'] for r in layer_6_results]
            
            colors = ['red' if n == 87 else 'lightblue' for n in neurons]
            bars = ax4.bar(range(len(neurons)), ratios, color=colors)
            ax4.set_title('Layer 6 Neuron Specialization (Red = Neuron 87)')
            ax4.set_xlabel('Top Neurons (by rank)')
            ax4.set_ylabel('Specialization Ratio')
            
            # Add neuron numbers as labels
            for i, (bar, neuron) in enumerate(zip(bars, neurons)):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'N{neuron}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results, filename="specialization_results.csv"):
        """
        Save results to CSV file
        """
        rows = []
        for layer, layer_results in results.items():
            for result in layer_results:
                row = {'layer': layer, **result}
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        return df
    
    def analyze_layer_progression(self, results):
        """
        Analyze how specialization progresses through layers with improved metrics
        """
        print("\n" + "="*50)
        print("LAYER PROGRESSION ANALYSIS")
        print("="*50)
        
        layers = sorted(results.keys())
        
        for layer in layers:
            layer_results = results[layer]
            
            if not layer_results:
                continue
                
            max_effect_size = max(r['effect_size'] for r in layer_results)
            avg_effect_size = np.mean([r['effect_size'] for r in layer_results])
            max_ratio = max(r['specialization_ratio'] for r in layer_results)
            avg_ratio = np.mean([r['specialization_ratio'] for r in layer_results])
            
            print(f"\nLayer {layer}:")
            print(f"  Max effect size: {max_effect_size:.2f}")
            print(f"  Avg effect size: {avg_effect_size:.2f}")
            print(f"  Max specialization ratio: {max_ratio:.2f}x")
            print(f"  Avg specialization ratio: {avg_ratio:.2f}x")
            
            # More conservative threshold for "high specialization"
            if max_effect_size > 2.0:  # Large effect size
                print(f"  🔥 HIGH SPECIALIZATION DETECTED (Effect Size > 2.0)!")
            elif max_effect_size > 0.8:  # Medium effect size
                print(f"  ⚡ MODERATE SPECIALIZATION (Effect Size > 0.8)")
            
            # Highlight neuron 87 in layer 6 if present
            if layer == 6:
                neuron_87_result = next((r for r in layer_results if r['neuron'] == 87), None)
                if neuron_87_result:
                    print(f"  🎯 Neuron 87: Effect size {neuron_87_result['effect_size']:.2f}, "
                          f"Ratio {neuron_87_result['specialization_ratio']:.2f}x")
                    rank = next(i for i, r in enumerate(layer_results) if r['neuron'] == 87) + 1
                    print(f"      Rank: {rank} out of top 10")
                else:
                    print(f"  ❌ Neuron 87 not in top 10 specialized neurons")
    
    def diagnostic_analysis(self, virtue_statements, neutral_statements):
        """
        Perform diagnostic analysis to check for potential issues
        """
        print("\n" + "="*50)
        print("DIAGNOSTIC ANALYSIS")
        print("="*50)
        
        # Check dataset characteristics
        print(f"Dataset sizes: Virtue={len(virtue_statements)}, Neutral={len(neutral_statements)}")
        
        # Sample some statements
        print("\nSample virtue statements:")
        for i, stmt in enumerate(virtue_statements[:3]):
            print(f"  {i+1}. {stmt[:100]}...")
        
        print("\nSample neutral statements:")
        for i, stmt in enumerate(neutral_statements[:3]):
            print(f"  {i+1}. {stmt[:100]}...")
        
        # Check text lengths
        virtue_lengths = [len(stmt.split()) for stmt in virtue_statements]
        neutral_lengths = [len(stmt.split()) for stmt in neutral_statements]
        
        print(f"\nText length statistics:")
        print(f"  Virtue - Mean: {np.mean(virtue_lengths):.1f}, Std: {np.std(virtue_lengths):.1f}")
        print(f"  Neutral - Mean: {np.mean(neutral_lengths):.1f}, Std: {np.std(neutral_lengths):.1f}")
        
        # Quick activation check for layer 6
        self.register_hooks([6])
        
        print("\nQuick layer 6 activation check:")
        virtue_acts = self.get_activations(virtue_statements[:5])
        neutral_acts = self.get_activations(neutral_statements[:5])
        
        if 'layer_6' in virtue_acts and 'layer_6' in neutral_acts:
            v_mean = virtue_acts['layer_6'].mean()
            n_mean = neutral_acts['layer_6'].mean()
            v_std = virtue_acts['layer_6'].std()
            n_std = neutral_acts['layer_6'].std()
            
            print(f"  Overall activation - Virtue: {v_mean:.4f}±{v_std:.4f}, Neutral: {n_mean:.4f}±{n_std:.4f}")
            
            # Check neuron 87 specifically
            if virtue_acts['layer_6'].shape[1] > 87:
                n87_virtue = virtue_acts['layer_6'][:, 87].mean()
                n87_neutral = neutral_acts['layer_6'][:, 87].mean()
                print(f"  Neuron 87 - Virtue: {n87_virtue:.4f}, Neutral: {n87_neutral:.4f}")
        
        self.clear_hooks()


def main():
    # Initialize analyzer
    analyzer = VirtueEthicsAnalyzer()
    
    # Load datasets
    virtue_statements, neutral_statements = analyzer.load_datasets(
        "data/virtue_top20.txt", 
        "data/neutral_statements.txt"
    )
    
    # Run diagnostic analysis first
    analyzer.diagnostic_analysis(virtue_statements, neutral_statements)
    
    # Analyze specialization across all layers with improved methodology
    print("Analyzing neural specialization across layers...")
    results = analyzer.analyze_specialization(
        virtue_statements, 
        neutral_statements, 
        target_layers=list(range(12)),  # All 12 layers
        top_k=10,
        min_activation_threshold=0.01  # Filter out very low activations
    )
    
    # Analyze layer progression
    analyzer.analyze_layer_progression(results)
    
    # Save results
    df = analyzer.save_results(results)
    
    # Create visualizations (you may want to update this based on new metrics)
    analyzer.visualize_specialization(results)
    
    # Clean up
    analyzer.clear_hooks()
    
    print("\nAnalysis complete!")
    print("\nRecommendations:")
    print("1. Check if effect sizes are more reasonable than ratios")
    print("2. Look for neuron 87 in layer 6 results")
    print("3. Compare virtue vs neutral statement characteristics")
    print("4. Consider using a more similar baseline dataset")
    
    return results, df

if __name__ == "__main__":
    results, df = main()