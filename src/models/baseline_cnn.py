import torch
import torch.nn as nn
import torchaudio

class LogMelCNNBaseline(nn.Module):
    """
    Initial log-mel spectrogram baseline with a 2D CNN acoustic encoder mapping to a temporal classifier[cite: 1].
    """
    def __init__(self, sample_rate=16000, n_mels=80, num_classes=7):
        super().__init__()
        
        # Spectrogram parameters[cite: 1]
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400, # 25ms window[cite: 1]
            hop_length=160, # 10ms hop[cite: 1]
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Small/medium CNN for local time-frequency patterns[cite: 1]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Temporal downsampling to approximately 100 ms temporal tokens (pool temporal dim by 2-3)[cite: 1]
            nn.MaxPool2d(kernel_size=(4, 2)) 
        )
        
        # Linear classifier: Multi-label classification for overlapping events[cite: 1]
        cnn_out_dim = 128 * (n_mels // 16)
        self.classifier = nn.Linear(cnn_out_dim, num_classes)

    def forward(self, x):
        # x shape: [B, Samples]
        mel = self.mel_spectrogram(x)
        log_mel = self.amplitude_to_db(mel) 
        
        # Shape: [B, 1, Mels, Time_10ms]
        log_mel = log_mel.unsqueeze(1) 
        
        # Shape: [B, Channels, Mels_down, Time_100ms]
        cnn_out = self.cnn(log_mel) 
        
        B, C, F, T = cnn_out.size()
        
        # Shape: [B, T, C*F] - reshaped for per-time-step temporal classification[cite: 1]
        cnn_out = cnn_out.view(B, T, C * F)
        
        # Shape: [B, T, Classes] -> P(class c is active at time t)[cite: 1]
        logits = self.classifier(cnn_out) 
        
        return logits