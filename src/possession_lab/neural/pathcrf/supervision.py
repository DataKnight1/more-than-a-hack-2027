"""Loss functions and multi-task supervision for N020 PathCRF.

Computes:
1. Dead-ball binary loss (BCEWithLogits)
2. Conditional Home/Away team loss (CrossEntropy on alive frames)
3. Unary edge emission cross-entropy (where strong path labels are known)
4. CRF sequence negative log-likelihood (on verified legal segments)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from possession_lab.neural.canonical import D, X

from .backbone import N020Outputs
from .crf import DynamicSparseCRF


@dataclass
class LossWeights:
    """Configurable loss weighting coefficients."""

    w_dead: float = 1.0
    w_team: float = 1.0
    w_edge: float = 0.5
    w_crf: float = 0.5
    w_reg: float = 1e-4


def compute_n020_loss(
    outputs: N020Outputs,
    labels: torch.Tensor,
    label_valid: torch.Tensor,
    path_labels: torch.Tensor | None = None,
    path_valid: torch.Tensor | None = None,
    crf: DynamicSparseCRF | None = None,
    weights: LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the multi-task composite loss for a batch.

    Args:
        outputs: N020Outputs from PathCRFNet.
        labels: [B, T] integer target codes (0=H, 1=A, 2=D, 3=X).
        label_valid: [B, T] boolean mask of valid graded supervision frames.
        path_labels: [B, T] optional target edge state indices [0..575].
        path_valid: [B, T] boolean mask of trustworthy path labels.
        crf: DynamicSparseCRF instance (optional).
        weights: LossWeights coefficients.

    Returns:
        total_loss: scalar differentiable tensor.
        loss_dict: dictionary of float metrics for logging.
    """
    # The explicit model/state-machine structure keeps the domain cases auditable.
    # pylint: disable=too-many-branches,too-many-statements
    if weights is None:
        weights = LossWeights()

    loss_dict: dict[str, float] = {}
    total_loss = torch.tensor(0.0, device=outputs.dead_logit.device)

    # 1. Dead-Ball Loss (BCEWithLogits)
    if label_valid.any():
        val_mask = label_valid
        is_dead = (labels == D).float()

        loss_dead = F.binary_cross_entropy_with_logits(
            outputs.dead_logit[val_mask],
            is_dead[val_mask],
        )
        total_loss = total_loss + weights.w_dead * loss_dead
        loss_dict["loss_dead"] = float(loss_dead.item())

        with torch.no_grad():
            pred_dead = (outputs.dead_logit[val_mask] > 0.0).float()
            acc_dead = (pred_dead == is_dead[val_mask]).float().mean().item()
            loss_dict["acc_dead"] = float(acc_dead)
    else:
        loss_dict["loss_dead"] = 0.0
        loss_dict["acc_dead"] = 0.0

    # 2. Conditional Home/Away Loss (CrossEntropy on alive frames)
    alive_mask = label_valid & (labels != D) & (labels != X)
    if alive_mask.any():
        target_team = labels[alive_mask]  # 0 for Home, 1 for Away
        alive_logits = outputs.team_logits[alive_mask]

        loss_team = F.cross_entropy(alive_logits, target_team)
        total_loss = total_loss + weights.w_team * loss_team
        loss_dict["loss_team"] = float(loss_team.item())

        with torch.no_grad():
            pred_team = alive_logits.argmax(dim=-1)
            acc_team = (pred_team == target_team).float().mean().item()
            loss_dict["acc_team"] = float(acc_team)
    else:
        loss_dict["loss_team"] = 0.0
        loss_dict["acc_team"] = 0.0

    # 3. Supervised Edge Loss (if strong labels provided)
    if path_labels is not None and path_valid is not None:
        p_mask = path_valid & (path_labels >= 0)
        if p_mask.any():
            valid_targets = path_labels[p_mask]
            valid_logits = outputs.compressed_edge_logits[p_mask]
            loss_edge = F.cross_entropy(valid_logits, valid_targets)
            total_loss = total_loss + weights.w_edge * loss_edge
            loss_dict["loss_edge"] = float(loss_edge.item())

            with torch.no_grad():
                pred_edge = valid_logits.argmax(dim=-1)
                acc_edge = (pred_edge == valid_targets).float().mean().item()
                loss_dict["acc_edge"] = float(acc_edge)
        else:
            loss_dict["loss_edge"] = 0.0
    else:
        loss_dict["loss_edge"] = 0.0

    # 4. CRF Sequence NLL (if CRF model and full legal path sequence available)
    if crf is not None and path_labels is not None and path_valid is not None:
        # Check if batch has any sequences with fully valid paths
        full_valid = path_valid.all(dim=1) & (path_labels >= 0).all(dim=1)
        if full_valid.any():
            sub_emiss = outputs.compressed_edge_logits[full_valid]
            sub_embed = outputs.edge_embeds[full_valid]
            sub_tags = path_labels[full_valid]
            loss_crf = crf.neg_log_likelihood(sub_emiss, sub_embed, sub_tags, reduction="mean")
            if not torch.isfinite(loss_crf):
                raise ValueError("CRF loss is not finite")
            total_loss = total_loss + weights.w_crf * loss_crf
            loss_dict["loss_crf"] = float(loss_crf.item())
        else:
            loss_dict["loss_crf"] = 0.0
    else:
        loss_dict["loss_crf"] = 0.0

    loss_dict["loss_total"] = float(total_loss.item())
    return total_loss, loss_dict
