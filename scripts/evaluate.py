import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import VaaniWindowedDataset
from src.models.transformer_model import CNNTransformerModel  # Updated import
from src.decoding.decoder import EventDecoder
from src.evaluation.metrics import evaluate_predictions

def evaluate_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    hf_dataset = load_dataset("data/raw/Vaani-Noise-Event-Dataset", split="train")
    hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
    
    eval_subset = hf_dataset.select(range(1000))
    eval_dataset = VaaniWindowedDataset(eval_subset, tier_filter="Gold")
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Updated: Initialize Transformer and load its checkpoint
    model = CNNTransformerModel().to(device)
    checkpoint_path = "experiments/checkpoints/baseline_transformer_gold.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print("Loaded trained Transformer weights.")
    else:
        print("WARNING: Checkpoint not found. Evaluating with random weights.")
    model.eval()
    
    decoder = EventDecoder(threshold=0.5, min_duration=0.1)
    
    predictions_dict = {}
    ground_truth_dict = {}
    
    print("Running inference and decoding events...")
    with torch.no_grad():
        for batch_idx, (waveforms, targets) in enumerate(eval_loader):
            clip_id = f"window_{batch_idx}"
            
            logits = model(waveforms.to(device))
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            pred_events = decoder.decode(probs, window_duration=5.0)
            predictions_dict[clip_id] = {'events': pred_events}
            
            targets_numpy = targets.squeeze(0).numpy()
            ref_events = decoder.decode(targets_numpy, window_duration=5.0)
            for r in ref_events:
                r['start'] = r.pop('onset')
                r['end'] = r.pop('offset')
            ground_truth_dict[clip_id] = {'events': ref_events}

    print("\nCalculating metrics...")
    results = evaluate_predictions(predictions_dict, ground_truth_dict)
    
    print(f"=====================================")
    print(f" Event F1     : {results['Event_F1']:.4f}")
    print(f" Segment Dice : {results['Segment_Dice']:.4f}")
    print(f" Combined     : {results['Combined']:.4f}")
    print(f"=====================================")

if __name__ == "__main__":
    evaluate_transformer()