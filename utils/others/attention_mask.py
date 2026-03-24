import torch

def make_attention_mask(hidden_states: torch.Tensor, lengths: torch.Tensor):
    """
    Args:
        hidden_states: shape of (B, L, *)
        lengths: shape of (B)

    Returns: shape of (B, L)
    """

    batch_size, max_length = hidden_states.size(0), hidden_states.size(1)
    range_tensor = torch.arange(max_length, device=hidden_states.device).repeat(batch_size, 1)
    attention_mask = torch.as_tensor(range_tensor < lengths.unsqueeze(1), device=hidden_states.device)

    return attention_mask