import torch


def kl_loss(p, q):
    # p: (N, n_heads, L, L) q: (N, n_heads, L, L)
    logits = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))  # (N, n_heads, L, L)
    kl = torch.sum(logits, dim=-1)  # (N, n_heads, L)
    
    return torch.mean(kl, dim=1)  # (N, L)


def entropy_loss(p):
    # p: (N, n_heads, L, L)
    logits = -p * torch.log(p + 0.0001)
    entropy = torch.sum(logits, dim=-1)  # (N, n_heads, L)

    return torch.mean(entropy, dim=1)  # (N, L)