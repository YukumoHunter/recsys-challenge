import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self attention module, adapted from the un-official implementation of the GERL paper.

    Args:
        embedding_size (int): The size of the input embeddings.
        hidden_size (int): The size of the hidden layer.
        mask_filler (float): The value to fill the masked positions with. Defaults to -1e9.
    """

    def __init__(
        self, embedding_size: int, hidden_size: int, mask_filler: float = -1e9
    ):
        super(SelfAttention, self).__init__()
        self.mask_filler = mask_filler

        self.transform = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh(),
        )

        self.gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, seqs, seq_masks=None):
        gates = self.gate(self.transform(seqs)).squeeze(-1)
        if seq_masks:
            gates = gates.masked_fill(seq_masks, self.mask_filler)

        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)

        h = p_attn * seqs
        return h.sum(dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size: int, num_heads: int, head_size: int):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_size = head_size

        self.q_proj = nn.Linear(input_size, num_heads * head_size)
        self.k_proj = nn.Linear(input_size, num_heads * head_size)
        self.v_proj = nn.Linear(input_size, num_heads * head_size)

    def forward(self, key, value, query, mask=None):
        batch_size, seq_len, input_size = key.size()

        assert (
            input_size == self.input_size
        ), f"Input size mismatch: {input_size} != {self.input_size}"

        # project q k v to MHA space
        q = (
            self.q_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_size)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, -1, self.num_heads, self.head_size)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, -1, self.num_heads, self.head_size)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size**0.5)

        if mask:
            scores += mask

        p_attn = F.softmax(scores, dim=-1)

        return (
            torch.matmul(p_attn, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )


def create_mask_from_len_for_seqs(
    seq_lens: torch.Tensor, max_len: int
) -> torch.LongTensor:
    seqs = torch.arange(max_len, device=seq_lens.device, dtype=seq_lens.dtype).expand(
        seq_lens.size(0), max_len
    ) < seq_lens.unsqueeze(1)

    return seqs.long()
