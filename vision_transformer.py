from torch import nn
from torchaudio.transforms import MelSpectrogram


class ViT(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers):
        super().__init__()
        self.mel_spec = MelSpectrogram(sample_rate=16000, n_fft=2048, win_length=400, hop_length=160, n_mels=128)
        self.conv = nn.Conv2d(1, hidden_size, kernel_size=16, stride=16)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, dim_feedforward=hidden_size * 4, nhead=num_heads)
        self.layers = nn.TransformerEncoder(layer, num_layers)

    def forward(self, inputs):
        hidden_states = self.mel_spec(inputs)
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.view(hidden_states.size(0), hidden_states.size(1), -1)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.layers(hidden_states)

        return hidden_states
