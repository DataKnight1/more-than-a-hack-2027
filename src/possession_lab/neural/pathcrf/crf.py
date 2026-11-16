"""Dynamic Sparse Conditional Random Field over compressed possession edges.

Adapted from original PathCRF (Kim et al., KDD 2026, MPL-2.0).
Computes exact forward log-partition logZ, gold path scoring, NLL loss,
Viterbi decoding, and forward-backward state marginals over the legal 576-state graph.
"""

from __future__ import annotations

import torch
from torch import nn

from .transitions import build_incoming_structures

DEFAULT_FORBID_SCORE: float = -1e4


class DynamicSparseCRF(nn.Module):  # type: ignore[misc]
    """Dynamic Sparse CRF sequence model over K=576 compressed edge states.

    Scores path transitions dynamically using pairwise edge representations:
        trans_score(prev, curr) = W * [prev_embed; curr_embed] + b
    Transitions not present in the legal incoming list receive forbid_score (-10,000).
    """

    def __init__(
        self,
        comp_src: torch.Tensor,
        comp_dst: torch.Tensor,
        edge_embed_dim: int = 16,
        team_size: int = 11,
        forbid_score: float = DEFAULT_FORBID_SCORE,
    ) -> None:
        super().__init__()
        self.K = comp_src.numel()
        self.edge_embed_dim = edge_embed_dim
        self.team_size = team_size
        self.forbid_score = forbid_score

        self.register_buffer("comp_src", comp_src)
        self.register_buffer("comp_dst", comp_dst)

        # Transition MLP mapping concatenated pair [prev_embed; curr_embed] -> scalar score
        self.linear = nn.Linear(edge_embed_dim * 2, 1)

        # Precompute incoming adjacency buffers
        inc_idx, inc_type, inc_mask = build_incoming_structures(comp_src, comp_dst, team_size)
        self.register_buffer("inc_idx", inc_idx)  # [K, M]
        self.register_buffer("inc_type", inc_type)  # [K, M]
        self.register_buffer("inc_mask", inc_mask)  # [K, M]

    def _logsumexp_masked(self, x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Numerically stable logsumexp over masked candidates."""
        neg = torch.full_like(x, self.forbid_score)
        x_masked = torch.where(mask, x, neg)
        return torch.logsumexp(x_masked, dim=dim)

    def _compute_logZ(
        self,
        emissions: torch.Tensor,
        edge_embeds: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the total log-partition function logZ via the forward algorithm in float32.

        Args:
            emissions: [B, T, K] unary edge potentials.
            edge_embeds: [B, T, K, D] compressed edge representations.
            mask: [B, T] boolean validity mask.

        Returns:
            logZ: [B] total log partition value.
        """
        B, T, K = emissions.shape
        D = self.edge_embed_dim

        # Enforce float32 for sequence stability
        emissions = emissions.float()
        edge_embeds = edge_embeds.float()

        mask = emissions.new_ones((B, T), dtype=torch.bool) if mask is None else mask.bool()

        alpha = emissions[:, 0]  # [B, K]

        inc_idx: torch.Tensor = self.inc_idx.clamp(min=0)  # [K, M]
        inc_idx_alpha = inc_idx.unsqueeze(0).expand(B, -1, -1)  # [B, K, M]
        inc_idx_edge = inc_idx.view(1, K, -1, 1).expand(B, -1, -1, D)  # [B, K, M, D]
        inc_mask: torch.Tensor = self.inc_mask.unsqueeze(0).expand(B, -1, -1)  # [B, K, M]

        for t in range(1, T):
            emit_score = emissions[:, t]  # [B, K]

            prev_alpha = alpha.unsqueeze(1).expand(-1, K, -1)  # [B, K_curr, K_prev]
            prev_alpha = prev_alpha.gather(2, inc_idx_alpha)  # [B, K_curr, M]

            prev_embed = edge_embeds[:, t - 1]  # [B, K, D]
            curr_embed = edge_embeds[:, t]  # [B, K, D]
            prev_embed = prev_embed.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K_curr, K_prev, D]
            curr_embed = curr_embed.unsqueeze(2).expand(
                -1, -1, inc_idx.size(1), -1
            )  # [B, K_curr, M, D]

            prev_embed_gather = prev_embed.gather(2, inc_idx_edge)  # [B, K, M, D]
            trans_score = self.linear(torch.cat([prev_embed_gather, curr_embed], dim=-1)).squeeze(
                -1
            )  # [B, K, M]

            score = self._logsumexp_masked(prev_alpha + trans_score, inc_mask, dim=2)  # [B, K]
            new_alpha = score + emit_score  # [B, K]

            alpha = torch.where(mask[:, t].unsqueeze(1), new_alpha, alpha)

        logZ = torch.logsumexp(alpha, dim=1)  # [B]
        return logZ

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,
        edge_embeds: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute exact score for a target sequence of compressed edge tags.

        Args:
            emissions: [B, T, K] unary edge potentials.
            edge_embeds: [B, T, K, D] edge embeddings.
            tags: [B, T] target edge IDs in [0..K-1].
            mask: [B, T] boolean validity mask.

        Returns:
            gold_score: [B] scalar path score.
        """
        B, T, _ = emissions.shape
        D = self.edge_embed_dim

        emissions = emissions.float()
        edge_embeds = edge_embeds.float()

        mask = emissions.new_ones((B, T), dtype=torch.bool) if mask is None else mask.bool()

        inc_idx: torch.Tensor = self.inc_idx.clamp(min=0)  # [K, M]
        inc_mask: torch.Tensor = self.inc_mask  # [K, M]

        score = emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)  # [B]
        forbid_score_t = emissions.new_full((B,), self.forbid_score)

        for t in range(1, T):
            prev = tags[:, t - 1]
            curr = tags[:, t]
            emit_score = emissions[:, t].gather(1, curr.unsqueeze(1)).squeeze(1)  # [B]

            prev_embed = edge_embeds[:, t - 1]  # [B, K, D]
            curr_embed = edge_embeds[:, t]  # [B, K, D]
            prev_embed = prev_embed.gather(1, prev.view(B, 1, 1).expand(-1, 1, D)).squeeze(
                1
            )  # [B, D]
            curr_embed = curr_embed.gather(1, curr.view(B, 1, 1).expand(-1, 1, D)).squeeze(
                1
            )  # [B, D]
            trans_score = self.linear(torch.cat([prev_embed, curr_embed], dim=-1)).squeeze(
                -1
            )  # [B]

            inc_idx_t = inc_idx[curr]  # [B, M]
            inc_mask_t = inc_mask[curr]  # [B, M]
            is_allowed = ((inc_idx_t == prev.unsqueeze(1)) & inc_mask_t).any(dim=1)  # [B]
            trans_score = torch.where(is_allowed, trans_score, forbid_score_t)

            score = score + (emit_score + trans_score) * mask[:, t].float()

        return score

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        edge_embeds: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Negative log-likelihood of target edge sequences: NLL = logZ - gold_score."""
        logZ = self._compute_logZ(emissions, edge_embeds, mask)
        gold = self._compute_gold_score(emissions, edge_embeds, tags, mask)
        nll = logZ - gold
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        return nll

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def decode(
        self,
        emissions: torch.Tensor,
        edge_embeds: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Viterbi decoding of the highest-scoring legal possession path.

        Args:
            emissions: [B, T, K] unary edge potentials.
            edge_embeds: [B, T, K, D] edge embeddings.
            mask: [B, T] boolean mask.

        Returns:
            best_tags: [B, T] predicted edge state sequence.
        """
        B, T, K = emissions.shape
        D = self.edge_embed_dim

        emissions = emissions.float()
        edge_embeds = edge_embeds.float()

        mask = emissions.new_ones((B, T), dtype=torch.bool) if mask is None else mask.bool()

        inc_idx: torch.Tensor = self.inc_idx.clamp(min=0)  # [K, M]
        inc_idx_delta = inc_idx.unsqueeze(0).expand(B, -1, -1)  # [B, K, M]
        inc_idx_edge = inc_idx.view(1, K, -1, 1).expand(B, -1, -1, D)  # [B, K, M, D]
        inc_mask: torch.Tensor = self.inc_mask.unsqueeze(0).expand(B, -1, -1)  # [B, K, M]

        delta = emissions[:, 0]  # [B, K]
        forbid_score_t = emissions.new_full((B, K, inc_idx.size(1)), self.forbid_score)
        backpointers: list[torch.Tensor] = []

        for t in range(1, T):
            emit_score = emissions[:, t]  # [B, K]

            prev_delta = delta.unsqueeze(1).expand(-1, K, -1)  # [B, K_curr, K_prev]
            prev_delta = prev_delta.gather(2, inc_idx_delta)  # [B, K_curr, M]

            prev_embed = edge_embeds[:, t - 1]
            curr_embed = edge_embeds[:, t]
            prev_embed = prev_embed.unsqueeze(1).expand(-1, K, -1, -1)
            curr_embed = curr_embed.unsqueeze(2).expand(-1, -1, inc_idx.size(1), -1)

            prev_embed_gather = prev_embed.gather(2, inc_idx_edge)
            trans_score = self.linear(torch.cat([prev_embed_gather, curr_embed], dim=-1)).squeeze(
                -1
            )

            score = torch.where(inc_mask, prev_delta + trans_score, forbid_score_t)
            best_score, best_m = score.max(dim=2)  # [B, K], [B, K]

            new_delta = emit_score + best_score
            delta = torch.where(mask[:, t].unsqueeze(1), new_delta, delta)
            backpointers.append(best_m)

        best_last_tag = delta.argmax(dim=1)  # [B]
        best_tags = emissions.new_zeros((B, T), dtype=torch.long)
        best_tags[:, -1] = best_last_tag

        for t in reversed(range(1, T)):
            curr = best_tags[:, t]  # [B]
            best_m = backpointers[t - 1].gather(1, curr.unsqueeze(1)).squeeze(1)  # [B]
            prev = inc_idx[curr, best_m]  # [B]
            best_tags[:, t - 1] = prev

        return best_tags

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def compute_marginals(
        self,
        emissions: torch.Tensor,
        edge_embeds: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute exact posterior state marginal probabilities q_t(z) via forward-backward.

        Returns:
            marginals: [B, T, K] normalized state posterior probabilities; each row sums to 1.
        """
        B, T, K = emissions.shape
        D = self.edge_embed_dim

        emissions = emissions.float()
        edge_embeds = edge_embeds.float()

        mask = emissions.new_ones((B, T), dtype=torch.bool) if mask is None else mask.bool()

        # 1. Forward pass: compute alpha[t, k] = log P(x_{1:t}, z_t = k)
        alpha = torch.empty((B, T, K), dtype=torch.float32, device=emissions.device)
        alpha[:, 0] = emissions[:, 0]

        inc_idx: torch.Tensor = self.inc_idx.clamp(min=0)
        inc_idx_alpha = inc_idx.unsqueeze(0).expand(B, -1, -1)
        inc_idx_edge = inc_idx.view(1, K, -1, 1).expand(B, -1, -1, D)
        inc_mask: torch.Tensor = self.inc_mask.unsqueeze(0).expand(B, -1, -1)

        # Precompute transition scores trans_scores[t]: [B, K_curr, M]
        all_trans: list[torch.Tensor] = []
        for t in range(1, T):
            prev_embed = edge_embeds[:, t - 1].unsqueeze(1).expand(-1, K, -1, -1)
            curr_embed = edge_embeds[:, t].unsqueeze(2).expand(-1, -1, inc_idx.size(1), -1)
            prev_gather = prev_embed.gather(2, inc_idx_edge)
            trans_score = self.linear(torch.cat([prev_gather, curr_embed], dim=-1)).squeeze(-1)
            all_trans.append(trans_score)

            prev_alpha = alpha[:, t - 1].unsqueeze(1).expand(-1, K, -1).gather(2, inc_idx_alpha)
            score = self._logsumexp_masked(prev_alpha + trans_score, inc_mask, dim=2)
            alpha[:, t] = score + emissions[:, t]

        logZ = torch.logsumexp(alpha[:, -1], dim=-1, keepdim=True)  # [B, 1]

        # 2. Backward pass: compute beta[t, k] = log P(x_{t+1:T} | z_t = k)
        beta = torch.empty((B, T, K), dtype=torch.float32, device=emissions.device)
        beta[:, -1] = 0.0

        for t in reversed(range(T - 1)):
            trans_score = all_trans[t]  # [B, K_curr, M]
            emit_next = emissions[:, t + 1]  # [B, K_curr]
            beta_next = beta[:, t + 1]  # [B, K_curr]

            term = (emit_next + beta_next).unsqueeze(2) + trans_score  # [B, K_curr, M]
            term_masked = torch.where(inc_mask, term, torch.full_like(term, self.forbid_score))

            # Scatter-logsumexp back into state k_prev
            b_t = torch.full(
                (B, K), self.forbid_score, dtype=torch.float32, device=emissions.device
            )
            # Flatten across K_curr and M
            flat_term = term_masked.view(B, -1)
            flat_idx = inc_idx.unsqueeze(0).expand(B, -1, -1).contiguous().view(B, -1)

            # Accumulate per prev state
            for k in range(K):
                k_mask = flat_idx == k
                if k_mask.any():
                    vals = torch.where(
                        k_mask, flat_term, torch.full_like(flat_term, self.forbid_score)
                    )
                    b_t[:, k] = torch.logsumexp(vals, dim=-1)

            beta[:, t] = b_t

        # 3. Posterior marginals: log q_t(k) = alpha[t, k] + beta[t, k] - logZ
        log_marginals = alpha + beta - logZ.unsqueeze(1)
        marginals = torch.softmax(log_marginals, dim=-1)
        return marginals
