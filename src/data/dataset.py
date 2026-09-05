import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import json
import io

class VaaniWindowedDataset(Dataset):
    """
    Loads Vaani dataset, filters by tier, and chunks audio into fixed windows[cite: 1].
    """
    def __init__(self, hf_dataset, tier_filter="Gold", sample_rate=16000, 
                 window_sec=5.0, hop_sec=2.5, frame_duration=0.01):
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.frame_duration = frame_duration
        self.num_classes = 7
        
        self.class_map = {
            "animal": 0, "vehicle": 1, "traffic": 1, "baby": 2, "child": 2,
            "music": 3, "singing": 3, "phone": 4, "alarm": 4, "signal": 4,
            "appliance": 5, "machine": 5, "human": 6
        }
        
        self.windows = []
        self._prepare_windows(hf_dataset, tier_filter)

    def _map_category(self, raw_category):
        raw_lower = str(raw_category).lower()
        for key, idx in self.class_map.items():
            if key in raw_lower:
                return idx
        return -1 

    def _prepare_windows(self, hf_dataset, tier_filter):
        tier_mapping = {
            "Gold": "verified_timestamps",
            "Silver": "unverified_timestamps",
            "Bronze": "no_timestamps"
        }
        target_quality = tier_mapping.get(tier_filter)
        
        for record_idx, record in enumerate(hf_dataset):
            if record.get('annotationQuality') != target_quality:
                continue
                
            duration = float(record.get('duration', 0))
            if duration <= 0:
                continue
                
            events = []
            raw_timestamps = record.get('NoiseSubCategoryTimeStamp')
            if raw_timestamps:
                if isinstance(raw_timestamps, str):
                    try:
                        raw_timestamps = json.loads(raw_timestamps)
                    except:
                        raw_timestamps = []
                
                for ev in raw_timestamps:
                    cat_idx = self._map_category(record.get('NoiseCategory', ''))
                    if cat_idx != -1:
                        events.append({
                            'start': float(ev.get('start', 0)),
                            'end': float(ev.get('end', 0)),
                            'class_idx': cat_idx
                        })

            num_windows = max(1, int(np.ceil((duration - self.window_sec) / self.hop_sec)) + 1)
            audio_info = record.get('audio', {})
            
            for i in range(num_windows):
                w_start = i * self.hop_sec
                w_end = w_start + self.window_sec
                
                window_events = []
                for ev in events:
                    if ev['start'] < w_end and ev['end'] > w_start:
                        w_ev_start = max(0.0, ev['start'] - w_start)
                        w_ev_end = min(self.window_sec, ev['end'] - w_start)
                        window_events.append({
                            'start': w_ev_start,
                            'end': w_ev_end,
                            'class_idx': ev['class_idx']
                        })
                        
                self.windows.append({
                    'audio_path': audio_info.get('path'),
                    'audio_bytes': audio_info.get('bytes'),
                    'w_start': w_start,
                    'w_end': min(w_end, duration),
                    'events': window_events
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window_info = self.windows[idx]
        
        start_frame = int(window_info['w_start'] * self.sample_rate)
        num_frames = int((window_info['w_end'] - window_info['w_start']) * self.sample_rate)
        
        # CRITICAL FIX: Prioritize bytes over the path string for Hugging Face Parquet datasets
        if window_info.get('audio_bytes') is not None:
            audio_source = io.BytesIO(window_info['audio_bytes'])
        else:
            audio_source = window_info['audio_path']
            
        try:
            # We explicitly specify format="wav" to help torchaudio decode from memory
            waveform, sr = torchaudio.load(audio_source, frame_offset=start_frame, num_frames=num_frames, format="wav")
        except Exception:
            # Fallback if the backend does not support frame_offset on file-like objects
            if hasattr(audio_source, 'seek'):
                audio_source.seek(0)
            waveform, sr = torchaudio.load(audio_source, format="wav")
            waveform = waveform[:, start_frame:start_frame+num_frames]
        
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        target_length = int(self.window_sec * self.sample_rate)
        if waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        total_frames = int(self.window_sec / self.frame_duration)
        targets = torch.zeros((total_frames, self.num_classes), dtype=torch.float32)
        
        for ev in window_info['events']:
            start_f = max(0, int(ev['start'] / self.frame_duration))
            end_f = min(total_frames, int(ev['end'] / self.frame_duration))
            targets[start_f:end_f, ev['class_idx']] = 1.0
            
        return waveform.squeeze(0), targets