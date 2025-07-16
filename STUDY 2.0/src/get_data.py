"""
Improved method for generating controlled datasets of virtue ethics and neutral statements using GPT-2. 
Uses GPT-2 directly to produce statements to ensure high relevance for subsequent analysis of GPT-2 neuron activations
It ensures that statements are comparable in syntactic structure and length to allow for meaningful comparison of reasoning patterns.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import re
from tqdm import tqdm
import json

# Configuration
NUM_STATEMENTS = 100  # Per category
MAX_RETRIES = 3       # Per statement
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# Prompt templates designed to elicit comparable structures
VIRTUE_PROMPTS = [
    "From the perspective of virtue ethics, a good action is when someone",
    "Aristotle would praise someone for",
    "A virtuous person demonstrates their character by",
    "Moral excellence is shown through actions like"
]

NEUTRAL_PROMPTS = [
    "A factual description of everyday life would be",
    "An objective observation about the world is that",
    "A neutral statement about human behavior is that",
    "A non-judgmental description of an activity would be"
]

def generate_statements(prompt_templates, category, length_control=True):
    """Generate controlled statements using GPT-2"""
    statements = set()
    progress = tqdm(total=NUM_STATEMENTS, desc=f"Generating {category} statements")
    
    while len(statements) < NUM_STATEMENTS:
        # Rotate through prompt templates for diversity
        prompt = np.random.choice(prompt_templates)
        
        # Generate with controlled parameters
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            **inputs,
            max_new_tokens=15 if length_control else 30,
            num_return_sequences=5,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Process outputs
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            
            # Clean and standardize
            text = re.sub(r'^.*?:', '', text)  # Remove prompt prefix
            text = text.split('.')[0].strip()  # Take first complete sentence
            text = text[0].upper() + text[1:]  # Capitalize
            
            # Length matching filter (10-25 words)
            if length_control and not (10 <= len(text.split()) <= 25):
                continue
                
            # Content filters
            if (len(text) > 10 and 
                not any(w in text.lower() for w in ['however', 'but', 'although']) and
                not text.endswith(('?', '!'))):
                statements.add(text)
                progress.update(1)
                
                if len(statements) >= NUM_STATEMENTS:
                    break
    
    progress.close()
    return sorted(statements, key=lambda x: len(x))

def save_dataset(statements, filename):
    """Save statements with metadata"""
    metadata = {
        "generation_parameters": {
            "model": "gpt2",
            "num_statements": len(statements),
            "avg_length": np.mean([len(s.split()) for s in statements])
        },
        "statements": statements
    }
    
    # Save as both human-readable and JSON
    with open(filename, 'w') as f:
        f.write("\n".join(statements))
    
    with open(f"{filename.rsplit('.', 1)[0]}_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def verify_datasets(virtue, neutral):
    """Quality control checks"""
    # Length comparison
    v_lens = [len(s.split()) for s in virtue]
    n_lens = [len(s.split()) for s in neutral]
    
    print(f"\nQuality Control Report:")
    print(f"Virtue avg length: {np.mean(v_lens):.1f} words")
    print(f"Neutral avg length: {np.mean(n_lens):.1f} words")
    print(f"Vocabulary overlap: {len(set(' '.join(virtue).lower().split()) & set(' '.join(neutral).lower().split()))} shared words")

if __name__ == "__main__":
    print("Generating virtue ethics statements...")
    virtue_statements = generate_statements(VIRTUE_PROMPTS, "virtue ethics")
    save_dataset(virtue_statements, "virtue_ethics_gpt2.txt")
    
    print("\nGenerating neutral statements...")
    neutral_statements = generate_statements(NEUTRAL_PROMPTS, "neutral", length_control=True)
    save_dataset(neutral_statements, "neutral_statements_gpt2.txt")
    
    verify_datasets(virtue_statements, neutral_statements)
    print("\nDataset generation complete!")