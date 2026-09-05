import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
import os
import sys
import gc  # Added for explicit garbage collection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import VaaniWindowedDataset
from src.models.transformer_model import CNNTransformerModel

def train_baseline_2_sharded():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading local Vaani dataset (Memory-Mapped on disk)...")
    hf_dataset = load_dataset("data/raw/Vaani-Noise-Event-Dataset", split="train")
    hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
    
    # 1. Define how many chunks to split the dataset into.
    # Adjust this number to hit your ~5GB target based on your total dataset size.
    num_shards = 5 
    
    model = CNNTransformerModel().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    num_epochs = 30 # Increased epochs for standard Transformer training
    
    print(f"Starting Sharded Training: {num_shards} shards per epoch.")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0
        
        # 2. Iterate through the dataset one shard at a time
        for shard_idx in range(num_shards):
            print(f"\n--- Epoch [{epoch+1}/{num_epochs}] | Loading Shard {shard_idx+1}/{num_shards} ---")
            
            # Extract just this chunk from the disk-backed Hugging Face dataset
            shard = hf_dataset.shard(num_shards=num_shards, index=shard_idx)
            
            # Build the windowed dataset and PyTorch loader ONLY for this shard
            train_dataset = VaaniWindowedDataset(shard, tier_filter="Gold")
            # Added num_workers=4 for faster data loading if your CPU supports it
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
            
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
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                running_loss += loss.item()
                total_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Shard {shard_idx+1} Batch {batch_idx} Loss: {loss.item():.4f}")
            
            # 3. Explicitly free up RAM before loading the next shard
            del train_dataset
            del train_loader
            del shard
            gc.collect()
                
        epoch_loss = running_loss / total_batches if total_batches > 0 else 0
        print(f"\n=== Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f} ===")
        
    os.makedirs("experiments/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "experiments/checkpoints/baseline_transformer_gold.pt")
    print("Model saved to experiments/checkpoints/baseline_transformer_gold.pt")

if __name__ == "__main__":
    train_baseline_2_sharded()