import torch
from torch import nn

from audioencoder import EncoderWithSpectrogram, Config as EncoderConfig

from pydantic import BaseModel

class Config(BaseModel):
    hidden_size: int
    num_classes: int
    max_length: int
    dropout_rate: float

class Model(nn.Module):
    def __init__(self, encoder_config: EncoderConfig, config: Config):
        super().__init__()
        self.encoder = EncoderWithSpectrogram(encoder_config, config.hidden_size)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, dim_feedforward=config.hidden_size * 4, nhead=4,
            batch_first=True
        )

    def forward(self, inputs, input_lengths):
        hidden_states, input_lengths = self.encoder(inputs, input_lengths)
        hidden_states = self.transformer(hidden_states)

        return hidden_states


