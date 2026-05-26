import random
from datasets import load_dataset

def generate_synthetic_data(num_samples=100, seed=42):
    """
    Generates a simple synthetic dataset in 'Adjective + Noun + Verb' format
    for compositional generalization experiments. The seed ensures reproducibility.
    """
    random.seed(seed)
    
    adjectives_pos = ["good", "great", "nice", "clever"]
    adjectives_neg = ["bad", "terrible", "poor", "dull"]
    nouns = ["student", "code", "model", "paper", "idea"]
    verbs = ["works", "runs", "fails", "helps"]

    data = []
    for _ in range(num_samples):
        label = random.choice([0, 1])
        adj = random.choice(adjectives_pos) if label == 1 else random.choice(adjectives_neg)
        noun = random.choice(nouns)
        verb = random.choice(verbs)
        
        sentence = f"{adj} {noun} {verb}"
        data.append((sentence, label))
        
    return data

def load_sst2_subset(num_samples=100, seed=42):
    """
    Pulls a small subset from the SST-2 dataset and shuffles it using the seed.
    Applies a length filter to avoid circuit explosion.
    """
    dataset = load_dataset("glue", "sst2")
    
    filtered_data = [x for x in dataset['train'] if len(x['sentence'].split()) <= 5]
    
    random.seed(seed)
    random.shuffle(filtered_data)
    
    data = [(x['sentence'].lower(), x['label']) for x in filtered_data[:num_samples]]
    
    return data

def get_dataset(source="synthetic", num_samples=100, seed=42):
    if source == "synthetic":
        data = generate_synthetic_data(num_samples, seed)
        return data, []
    elif source == "sst2":
        data = load_sst2_subset(num_samples, seed)
        return data, []
    else:
        raise ValueError("Source must be 'synthetic' or 'sst2'")

if __name__ == "__main__":
    # Quick test
    train, _ = get_dataset("synthetic", num_samples=10, seed=100)
    print("Synthetic Data Sample (Seed 100):", train[:3])