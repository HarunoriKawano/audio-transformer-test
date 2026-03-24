import torch
from torch import nn

from audioencoder import LongAudioEncoder, Config as EncoderConfig, Preprocessor

from pydantic import BaseModel

class Config(BaseModel):
    hidden_size: int
    num_classes: int
    max_length: int
    dropout_rate: float

class Model(nn.Module):
    def __init__(self, encoder_config: EncoderConfig, config: Config):
        super().__init__()
        self.preprocessor = Preprocessor(encoder_config)
        self.encoder = LongAudioEncoder(encoder_config)
        self.linear = nn.Linear(encoder_config.hidden_size, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_length, config.hidden_size))
        self.transformer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, dim_feedforward=config.hidden_size * 4, nhead=4,
            batch_first=True
        )
        self.decoder = Decoder(config.hidden_size, config.hidden_size, config.num_classes, config.dropout_rate)

    def forward(self, inputs, input_lengths):
        hidden_states, input_lengths, _ = self.preprocessor(inputs, input_lengths)
        hidden_states, input_lengths = self.encoder(hidden_states, input_lengths)
        hidden_states = self.linear(hidden_states)
        hidden_states = hidden_states + self.pos_embed[:, :hidden_states.size(1), :]
        seq = torch.arange(hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
        mask = seq >= input_lengths.unsqueeze(1)
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=mask)

        mask = (torch.arange(hidden_states.size(1), device=hidden_states.device) < input_lengths.unsqueeze(1)).unsqueeze(2)
        len_clamped = input_lengths.view(-1, 1).clamp(min=1)
        max_pool = hidden_states.masked_fill(~mask, -float('inf')).max(dim=1)[0]
        mean_pool = hidden_states.masked_fill(~mask, 0.0).sum(dim=1) / len_clamped
        var = ((hidden_states - mean_pool.unsqueeze(1))**2).masked_fill(~mask, 0.0).sum(dim=1) / (len_clamped - 1).clamp(min=1e-5)
        std_pool = torch.sqrt(var + 1e-8)

        # mean_pool = hidden_states.mean(dim=1)
        # max_pool, _ = hidden_states.max(dim=1)
        # std_pool = hidden_states.std(dim=1)
        hidden_states = torch.cat([mean_pool, max_pool, std_pool], dim=1)
        out = self.decoder(hidden_states)

        return out


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate):
        super().__init__()
        self.linear = nn.Linear(input_size*3, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        out = self.out(hidden_states)

        return out
