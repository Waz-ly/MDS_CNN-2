import torch
import numpy as np
from torchmetrics.functional.retrieval import retrieval_normalized_dcg

# https://github.com/tiianhk/timbremetrics/blob/main/timbremetrics/metrics.py

def _prepare_rank_scores(pred: torch.Tensor, true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrize matrices and strip diagonal self-comparisons."""
    pred = pred + pred.T
    true = true + true.T
    mask_ = torch.ones_like(pred, dtype=torch.bool)
    mask_.fill_diagonal_(False)
    N = pred.shape[0]
    pred = pred[mask_].reshape(N, N - 1)
    true = true[mask_].reshape(N, N - 1)
    return pred, true

def mae_metric(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - true))

def ndcg_retrieve_sim(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    pred, true = _prepare_rank_scores(pred, true)
    pred = 1 - pred
    true = 1 - true
    ndcg_scores = torch.zeros(pred.shape[0], device=pred.device)
    for i in range(pred.shape[0]):
        # torchmetrics NDCG expects integer relevance scores for true labels
        ndcg_scores[i] = retrieval_normalized_dcg(pred[i], (true[i] * 100).long())
    return ndcg_scores.mean()

def _compute_triplet_agreement_one_anchor(
    pred: torch.Tensor, true: torch.Tensor, margin: float = 0.1
) -> torch.Tensor:
    true_diff = true.unsqueeze(0) - true.unsqueeze(1)
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    valid_mask = torch.abs(true_diff) > margin
    upper_triangle_mask = torch.triu(
        torch.ones_like(valid_mask, dtype=torch.bool), diagonal=1
    )
    valid_mask = valid_mask & upper_triangle_mask
    agreement_mask = (pred_diff * true_diff) > 0
    agreements = torch.sum(agreement_mask & valid_mask)
    valid_pairs = torch.sum(valid_mask)
    if valid_pairs == 0:
        return torch.tensor(0.0, device=pred.device)
    return agreements / valid_pairs

def triplet_agreement(pred: torch.Tensor, true: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    pred, true = _prepare_rank_scores(pred, true)
    scores = torch.zeros(pred.shape[0], device=pred.device)
    for i in range(pred.shape[0]):
        scores[i] = _compute_triplet_agreement_one_anchor(
            pred[i], true[i], margin=margin
        )
    return scores.mean()

def get_scale_mismatch(
        predicted_matrix: torch.Tensor,
        target_matrix: torch.Tensor,
    ):

    entry_mask = target_matrix > 1e-8

    with torch.no_grad():

        scale = predicted_matrix[entry_mask].mean() / target_matrix[entry_mask].mean()

    return scale

def stress_loss(
        predicted_matrix: torch.Tensor,
        target_matrix: torch.Tensor,
        rescale_type: str = "none",
        loss_type: str = "MAE"
    ) -> tuple[torch.Tensor, int]:
    """
    rescale_type options:
        "none" - no rescaling
        "match" - linearly scale target matrix to minimize mismatch with predicted
        "normalize" - normalize target and predicted to [0, 1]

    returns:
        metric: torch.Tensor - average computed metric
        n items: int - number of items used to compute averaged metric
    """

    entry_mask = target_matrix > 1e-8

    match rescale_type:
        case "none":

            adjusted_target_matrix = target_matrix
            adjusted_predicted_matrix = predicted_matrix

        case "normalize":

            target_min = target_matrix.min()
            target_max = target_matrix.max()
            adjusted_target_matrix = (target_matrix - target_min) / (target_max - target_min + 1e-8)

            pred_min = predicted_matrix.min()
            pred_max = predicted_matrix.max()
            adjusted_predicted_matrix = (predicted_matrix - pred_min) / (pred_max - pred_min + 1e-8)

        case _:

            raise Exception(f"rescale_type must be none, or normalize")

    match loss_type:
        case "MAE":

            loss_val = mae_metric(adjusted_predicted_matrix[entry_mask], adjusted_target_matrix[entry_mask])
            return loss_val, int(entry_mask.sum().item())

        case "triplet":

            loss_val = triplet_agreement(adjusted_predicted_matrix, adjusted_target_matrix)
            return loss_val, len(adjusted_predicted_matrix)
        
        case "NDCG":

            loss_val = ndcg_retrieve_sim(adjusted_predicted_matrix, adjusted_target_matrix)
            return loss_val, len(adjusted_predicted_matrix)

        case _:
        
            raise Exception(f"Invalid loss_type: {loss_type}")
