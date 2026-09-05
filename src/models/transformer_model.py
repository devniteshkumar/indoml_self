import torch
import torch.nn as nn
import torchaudio
import math

class PositionalEncoding(nn.Module):
    """
    Injects temporal order information into the sequence[cite: 1].
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CNNTransformerModel(nn.Module):
    """
    Log-mel -> CNN -> Transformer -> Per-time-step temporal classifier[cite: 1].
    """
    def __init__(self, sample_rate=16000, n_mels=80, num_classes=7, 
                 d_model=256, nhead=8, num_layers=4, dim_feedforward=1024):
        super().__init__()
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # 2D CNN Acoustic Encoder[cite: 1]
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
            nn.MaxPool2d(kernel_size=(4, 2)) 
        )
        
        cnn_out_dim = 128 * (n_mels // 16)
        
        # Project CNN output to Transformer embedding dimension[cite: 1]
        self.input_proj = nn.Linear(cnn_out_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)
        
        # Transformer Temporal Encoder (Pre-norm)[cite: 1]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1, 
            batch_first=True,
            norm_first=True # Pre-norm architecture[cite: 1]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-label classification head[cite: 1]
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        mel = self.mel_spectrogram(x)
        log_mel = self.amplitude_to_db(mel).unsqueeze(1) 
        
        cnn_out = self.cnn(log_mel) 
        B, C, F, T = cnn_out.size()
        
        # Reshape to [B, T, D] for temporal token sequence[cite: 1]
        cnn_out = cnn_out.view(B, T, C * F)
        
        x = self.input_proj(cnn_out)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        logits = self.classifier(x) 
        return logits