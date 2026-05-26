import music21
import json
import os

# Prefer SLURM_SUBMIT_DIR when running under sbatch; otherwise use cwd
PROJECT_ROOT = os.environ.get('SLURM_SUBMIT_DIR', os.getcwd())

def prepare_qnlp_music_dataset(max_chorales=100):
    bach_bundle = music21.corpus.search('bach')
    print(f"--- Found total Bach chorales: {len(bach_bundle)} ---")

    dataset = []
    processed_count = 0

    for result in bach_bundle:
        if processed_count >= max_chorales:
            break
            
        score = result.parse()
        
    # --- 1. LABELING (MAJOR / MINOR) ---
        try:
            key = score.analyze('key')
            if key.mode == 'major':
                interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch('C'))
                label = 1  # 1: Major (bright/happy)
            elif key.mode == 'minor':
                interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch('A'))
                label = 0  # 0: Minor (dark/sad)
            else:
                continue # Skip if neither major nor minor is detected
            
            transposed_score = score.transpose(interval)
        except Exception:
            continue

    # Extract main melody (Soprano)
        try:
            part = transposed_score.parts[0]
        except IndexError:
            continue
        
    # --- 2. CONVERT MEASURES TO NLP SENTENCES ---
        for measure in part.getElementsByClass(music21.stream.Measure):
            if measure.duration.quarterLength != 4.0:
                continue

            notes = measure.flatten().notes
            
            # Silence/rest check
            total_length = sum([n.duration.quarterLength for n in notes if hasattr(n, 'duration')])
            if total_length != 4.0:
                continue 
            
            sentence_words = []
            for element in notes:
                if isinstance(element, music21.note.Note):
                    # Create NLP token: "NoteName_Duration" (e.g. C5_quarter)
                    word = f"{element.pitch.nameWithOctave}_{element.duration.type}"
                    sentence_words.append(word)
                    
                elif isinstance(element, music21.chord.Chord):
                    highest = element.sortAscending().notes[-1]
                    word = f"{highest.pitch.nameWithOctave}_{highest.duration.type}"
                    sentence_words.append(word)
            
            if sentence_words:
                # Join tokens with spaces to form a text sentence
                sentence_str = " ".join(sentence_words)
                dataset.append((sentence_str, label))
        
        processed_count += 1
        if processed_count % 20 == 0:
            print(f"{processed_count} chorales processed...")

    print("\n--- [SUCCESS] ---")
    print(f"Total NLP sentences extracted: {len(dataset)}")
    
    # Print a sample to show how the NLP model will see the data
    if dataset:
        print(f"Sample Major/Minor data: {dataset[0]}")

    # Save dataset as list-of-tuples for lambeq
    out_path = os.path.join(PROJECT_ROOT, 'scripts', 'data_utils', 'bach_qnlp_sentiment_dataset.json')
    with open(out_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"[*] Saved dataset to: {out_path}")
        
    return dataset

# Run (for quick test, set max_chorales=20)
prepare_qnlp_music_dataset()