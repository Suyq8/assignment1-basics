import torch

def cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between predictions and targets.

    Args:
        pred (torch.Tensor): Predictions from the model.
        target (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: Computed cross-entropy loss.
    """
    max_pred = pred.max(dim=-1, keepdim=True).values
    pred = pred - max_pred
    log_sum_exp = pred.logsumexp(dim=-1, keepdim=True)
    loss = log_sum_exp - pred.gather(dim=-1, index=target.unsqueeze(-1))
    
    return loss.mean()