import torch
from torch import nn


class PositionalEncoder(nn.Module):

    def __init__(self, hidden_size: int, max_length: int):
        super().__init__()
        self.position_encoder = nn.Embedding(max_length, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`

        Returns:
            torch.Tensor: with shape `(1, L, D)`
        """
        max_length = hidden_states.size(1)
        position_ids = torch.arange(0, max_length, 1).to(hidden_states.device)
        position_embeddings = self.position_encoder(position_ids).unsqueeze(0)

        return position_embeddings


class RelativePositionEncoder(nn.Module):

    def __init__(self, hidden_size: int, max_length: int, with_cls: bool = False):
        super().__init__()
        self.max_length = max_length
        self.positional_embedding = nn.Embedding(self.max_length * 2 + 1, hidden_size)
        self.with_cls = with_cls
        self.cls_id = self.max_length * 2

    def forward(self, hidden_states: torch.Tensor):
        if self.with_cls:
            hidden_states = hidden_states[:, 1:]
        range_tensor = torch.arange(hidden_states.size(1), device=hidden_states.device)
        distance_mat = range_tensor[None, :] - range_tensor[:, None] + self.max_length


        if self.with_cls:
            distance_mat = torch.cat([torch.zeros_like(distance_mat[None, 0, :]) + self.cls_id, distance_mat], dim=0)
            distance_mat = torch.cat([torch.zeros_like(distance_mat[:, None, 0]) + self.cls_id, distance_mat], dim=1)

        position_embeddings = self.positional_embedding(distance_mat)
        return position_embeddings

