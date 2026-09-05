import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import VaaniWindowedDataset
from src.models.baseline_cnn import LogMelCNNBaseline

def train_gold_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading local Vaani dataset...")
    hf_dataset = load_dataset("data/raw/Vaani-Noise-Event-Dataset", split="train")
    
    # Disable automatic decoding to retrieve raw paths/bytes for torchaudio[cite: 1]
    hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
    
    train_dataset = VaaniWindowedDataset(hf_dataset, tier_filter="Gold")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    print(f"Generated {len(train_dataset)} overlapping windows for Gold tier.")
    
    model = LogMelCNNBaseline(sample_rate=16000, n_mels=80, num_classes=7).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    num_epochs = 5
    
    print("Starting Baseline 1 (CNN-only) Training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (waveforms, targets) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            logits = model(waveforms)
            
            # Downsample targets by pooling to match model output length[cite: 1]
            B, T_out, C = logits.shape
            targets_down = torch.nn.functional.adaptive_max_pool1d(
                targets.permute(0, 2, 1), output_size=T_out
            ).permute(0, 2, 1)
            
            loss = criterion(logits, targets_down)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f} ---")
        
    os.makedirs("experiments/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "experiments/checkpoints/baseline_cnn_gold.pt")
    print("Model saved to experiments/checkpoints/baseline_cnn_gold.pt")

if __name__ == "__main__":
    train_gold_baseline()