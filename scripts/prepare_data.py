from datasets import load_dataset
import os

def inspect_local_data():
    local_dir = "data/raw/Vaani-Noise-Event-Dataset"
    print(f"Loading dataset from local directory: {local_dir}...")
    
    # Load from the local parquet files downloaded by the hf CLI
    # We remove streaming=True since the files are now on disk
    dataset = load_dataset(local_dir, split="train")
    
    gold_count, silver_count, bronze_count = 0, 0, 0
    
    print("Inspecting first 50 files to verify annotation structure...\n")
    for idx, sample in enumerate(dataset):
        if idx == 0:
            print("--- Schema Discovery ---")
            print(f"Dataset keys: {list(sample.keys())}")
            print("------------------------\n")
            
        # 1. Map official schema to Gold/Silver/Bronze tiers
        quality = sample.get('annotationQuality', 'unknown')
        if quality == 'verified_timestamps':
            tier = 'Gold'
            gold_count += 1
        elif quality == 'unverified_timestamps':
            tier = 'Silver'
            silver_count += 1
        elif quality == 'no_timestamps':
            tier = 'Bronze'
            bronze_count += 1
        else:
            tier = 'unknown'
                
        # 2. Extract Event Timestamps
        events = sample.get('NoiseSubCategoryTimeStamp')
        if events is None:
            events = []
                
        # 3. Extract the audio ID
        audio_id = sample.get('imageFileName', f'audio_{idx}')
        if isinstance(audio_id, str):
            audio_id = os.path.basename(audio_id)
            
        print(f"[{idx}] ID: {audio_id} | Tier: {tier} | Events: {len(events)}")
        
        if idx >= 49:
            break
            
    print(f"\nInspection sample split: Gold: {gold_count}, Silver: {silver_count}, Bronze: {bronze_count}")

if __name__ == "__main__":
    inspect_local_data()