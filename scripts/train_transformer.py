import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import VaaniWindowedDataset
from src.models.transformer_model import CNNTransformerModel

def train_baseline_2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading local Vaani dataset...")
    hf_dataset = load_dataset("data/raw/Vaani-Noise-Event-Dataset", split="train")
    hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
    
    # Using Gold Tier only for Baseline 2[cite: 1]
    train_dataset = VaaniWindowedDataset(hf_dataset, tier_filter="Gold")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    
    print(f"Generated {len(train_dataset)} overlapping windows for Gold tier.")
    
    model = CNNTransformerModel().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4) # Slightly lower LR for Transformers
    
    num_epochs = 5
    
    print("Starting Baseline 2 (CNN + Transformer) Training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (waveforms, targets) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(waveforms)
            
            B, T_out, C = logits.shape
            targets_down = torch.nn.functional.adaptive_max_pool1d(
                targets.permute(0, 2, 1), output_size=T_out
            ).permute(0, 2, 1)
            
            loss = criterion(logits, targets_down)
            loss.backward()
            
            # Gradient clipping is highly recommended for Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f} ---")
        
    os.makedirs("experiments/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "experiments/checkpoints/baseline_transformer_gold.pt")
    print("Model saved to experiments/checkpoints/baseline_transformer_gold.pt")

if __name__ == "__main__":
    train_baseline_2()