"""Self-contained BTRIKcat (F4) architecture and controlled ablations.

This module needs only PyTorch and the Python standard library. All projection,
attention, pooling and regression-head components are defined here.
"""
from __future__ import annotations
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, expansion: int = 4):
        super().__init__()
        hidden = int(d_model * expansion)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1, ffn_expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)

        # ✅ FIX: Zero out PADDING QUERY before attention (critical!)
        if key_padding_mask is not None:
            h = h.masked_fill(~key_padding_mask.unsqueeze(-1), 0.0)

        attn_out, _ = self.attn(
            query=h, key=h, value=h,
            key_padding_mask=~key_padding_mask if key_padding_mask is not None else None,
            need_weights=False
        )

        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))

        # Final zero padding to prevent drift
        if key_padding_mask is not None:
            x = x.masked_fill(~key_padding_mask.unsqueeze(-1), 0.0)

        return x

class BiCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1, ffn_expansion: int = 4):
        super().__init__()
        self.p_norm_q = nn.LayerNorm(d_model)
        self.r_norm_kv = nn.LayerNorm(d_model)
        self.p_attends_r = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.r_norm_q = nn.LayerNorm(d_model)
        self.p_norm_kv = nn.LayerNorm(d_model)
        self.r_attends_p = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.p_norm_ffn = nn.LayerNorm(d_model)
        self.r_norm_ffn = nn.LayerNorm(d_model)
        self.p_ffn = FeedForward(d_model, dropout, ffn_expansion)
        self.r_ffn = FeedForward(d_model, dropout, ffn_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, protein: torch.Tensor, reaction: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None, reaction_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # --------------------------
        # Protein attends Reaction
        # --------------------------
        p_q = self.p_norm_q(protein)
        r_kv = self.r_norm_kv(reaction)

        # ✅ FIX: Zero padding queries
        if protein_mask is not None:
            p_q = p_q.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        if reaction_mask is not None:
            r_kv = r_kv.masked_fill(~reaction_mask.unsqueeze(-1), 0.0)

        p_delta, _ = self.p_attends_r(
            query=p_q, key=r_kv, value=r_kv,
            key_padding_mask=~reaction_mask if reaction_mask is not None else None,
            need_weights=False
        )

        # --------------------------
        # Reaction attends Protein
        # --------------------------
        r_q = self.r_norm_q(reaction)
        p_kv = self.p_norm_kv(protein)

        # ✅ FIX: Zero padding queries
        if reaction_mask is not None:
            r_q = r_q.masked_fill(~reaction_mask.unsqueeze(-1), 0.0)
        if protein_mask is not None:
            p_kv = p_kv.masked_fill(~protein_mask.unsqueeze(-1), 0.0)

        r_delta, _ = self.r_attends_p(
            query=r_q, key=p_kv, value=p_kv,
            key_padding_mask=~protein_mask if protein_mask is not None else None,
            need_weights=False
        )

        protein = protein + self.dropout(p_delta)
        reaction = reaction + self.dropout(r_delta)

        protein = protein + self.p_ffn(self.p_norm_ffn(protein))
        reaction = reaction + self.r_ffn(self.r_norm_ffn(reaction))

        # Zero padding
        if protein_mask is not None:
            protein = protein.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        if reaction_mask is not None:
            reaction = reaction.masked_fill(~reaction_mask.unsqueeze(-1), 0.0)

        return protein, reaction

class AttentionPool(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        weights = self.score(x)
        if mask is not None:
            # Keep the mask value representable under AMP (e.g. float16).
            weights = weights.masked_fill(
                ~mask.unsqueeze(-1), torch.finfo(weights.dtype).min
            )
        weights = torch.softmax(weights, dim=1)
        return torch.sum(weights * x, dim=1)

class _BTRIKcatBase(nn.Module):

    def __init__(self, protein_dim=2560, rxnfp_dim=256, chemberta_dim=384, d_model=256, num_heads=4, num_layers=1, self_layers=1, dropout=0, ffn_expansion=4, protein_max_tokens=128, reaction_max_tokens=17, pooling='attn_max', fusion='interaction', use_type_embeddings=True, reaction_modalities=('rxnfp', 'chemberta')):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling
        self.fusion = fusion
        self.use_type_embeddings = use_type_embeddings
        valid_modalities = {'rxnfp', 'chemberta'}
        self.reaction_modalities = tuple(reaction_modalities)
        if not self.reaction_modalities or not set(self.reaction_modalities) <= valid_modalities:
            raise ValueError('reaction_modalities must contain rxnfp and/or chemberta')
        self.protein_proj = nn.Sequential(nn.LayerNorm(protein_dim), nn.Linear(protein_dim, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, d_model))
        self.protein_local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.rxnfp_proj = nn.Sequential(nn.LayerNorm(rxnfp_dim), nn.Linear(rxnfp_dim, d_model))
        self.chemberta_proj = nn.Sequential(nn.LayerNorm(chemberta_dim), nn.Linear(chemberta_dim, d_model))
        self.protein_pos = nn.Parameter(torch.randn(1, protein_max_tokens, d_model) * 0.02)
        self.reaction_pos = nn.Parameter(torch.randn(1, reaction_max_tokens, d_model) * 0.02)
        if use_type_embeddings:
            self.protein_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.rxnfp_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.chemberta_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.protein_self_in = nn.ModuleList([SelfAttentionBlock(d_model, num_heads, dropout, ffn_expansion) for _ in range(self_layers)])
        self.reaction_self_in = nn.ModuleList([SelfAttentionBlock(d_model, num_heads, dropout, ffn_expansion) for _ in range(self_layers)])
        self.bi_cross_blocks = nn.ModuleList([BiCrossAttentionBlock(d_model, num_heads, dropout, ffn_expansion) for _ in range(num_layers)])
        self.protein_pooler = AttentionPool(d_model, dropout) if pooling in ('attn', 'attn_mean', 'attn_max') else None
        self.protein_post_pool_mean = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model)) if pooling == 'attn_mean' else None
        self.protein_post_pool_max = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model)) if pooling == 'attn_max' else None
        self.reaction_fusion_mlp = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.cls_norm = nn.LayerNorm(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.max_norm = nn.LayerNorm(d_model)
        self.reaction_fusion_mlp = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        single_pool_dim = d_model * 2 if pooling == 'max_mean' else d_model
        fused_dim = single_pool_dim * 4 if fusion == 'interaction' else single_pool_dim * 2
        self.gate = None
        if fusion in ('gated_sum', 'gated_concat'):
            self.gate = nn.Sequential(nn.Linear(single_pool_dim * 2, single_pool_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(single_pool_dim, single_pool_dim), nn.Sigmoid())
        hidden1 = max(d_model, fused_dim // 2)
        hidden2 = max(d_model // 2, hidden1 // 2)
        self.head = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, hidden1), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden1), nn.Linear(hidden1, hidden2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden2, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f'Expected 2D/3D tensor, got {x.shape}')

    @staticmethod
    def _resize_pos(pos: torch.Tensor, tgt_len: int) -> torch.Tensor:
        if pos.size(1) == tgt_len:
            return pos
        if tgt_len < pos.size(1):
            return pos[:, :tgt_len, :]
        dtype = pos.dtype
        pos = F.interpolate(pos.float().transpose(1, 2), size=tgt_len, mode='linear', align_corners=False).transpose(1, 2)
        return pos.to(dtype)

    def _pool_protein(self, x: torch.Tensor, pooler: Optional[AttentionPool], mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if self.pooling == 'mean':
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
                return x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return x.mean(1)
        if self.pooling == 'max':
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
            return x.max(1).values
        if self.pooling == 'max_mean':
            if mask is not None:
                x_max = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(1).values
                x_avg = x.masked_fill(~mask.unsqueeze(-1), 0.0).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                x_max = x.max(1).values
                x_avg = x.mean(1)
            return torch.cat([x_max, x_avg], -1)
        if self.pooling == 'attn':
            return pooler(x, mask)
        if self.pooling == 'attn_mean':
            attn = pooler(x, mask)
            if mask is not None:
                avg = x.masked_fill(~mask.unsqueeze(-1), 0.0).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                avg = x.mean(1)
            return self.protein_post_pool_mean(torch.cat([attn, avg], -1))
        if self.pooling == 'attn_max':
            attn = pooler(x, mask)
            if mask is not None:
                max_val = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(1).values
            else:
                max_val = x.max(1).values
            return self.protein_post_pool_max(torch.cat([attn, max_val], -1))

class BTRIKcat(_BTRIKcatBase):
    """F4 with token-level RXNFP-to-ChemBERTa attention and flexible protein input."""

    fusion_strategy_ablation = True

    def __init__(self, reaction_fusion_strategy: str = "attention", **kwargs) -> None:
        kwargs = dict(kwargs)
        kwargs.pop("reaction_fusion_strategy", None)
        if tuple(kwargs.get("reaction_modalities", ("rxnfp", "chemberta"))) != ("rxnfp", "chemberta"):
            raise ValueError("F4 uses both reaction inputs; select WithoutRXNFP/WithoutChemBERTa variants to remove a branch")
        if kwargs.get("pooling", "attn_max") not in {"mean", "max", "attn", "attn_mean", "attn_max"}:
            raise ValueError("Unsupported F4 pooling; max_mean has incompatible protein/reaction widths")
        if kwargs.get("fusion", "interaction") not in {"interaction", "concat", "gated_concat"}:
            raise ValueError("F4 fusion must be interaction, concat or gated_concat; gated_sum has incompatible head width")
        super().__init__(**kwargs)
        if reaction_fusion_strategy != "attention":
            raise ValueError("BTRIKcat fixes reaction_fusion_strategy='attention'")
        self.reaction_fusion_strategy = "attention"
        d_model = int(kwargs.get("d_model", 256))
        num_heads = int(kwargs.get("num_heads", 4))
        dropout = float(kwargs.get("dropout", 0.0))
        self.reaction_fusion_attn = torch.nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.reaction_fusion_norm = torch.nn.LayerNorm(d_model)

    def _fuse_reaction_tokens(
        self, rxnfp: torch.Tensor, chemberta: torch.Tensor
    ) -> torch.Tensor:
        if rxnfp.size(1) != 1:
            raise ValueError(
                "BTRIKcat expects one RXNFP global token, "
                f"got {rxnfp.size(1)}"
            )
        if chemberta.size(1) == 0:
            return rxnfp
        context = torch.cat([rxnfp, chemberta], dim=1)
        attended, _ = self.reaction_fusion_attn(
            query=chemberta,
            key=context,
            value=context,
            need_weights=False,
        )
        fused = self.reaction_fusion_norm(chemberta + attended)
        return torch.cat([rxnfp, fused], dim=1)

    def forward(
        self,
        protein_tokens: torch.Tensor,
        rxnfp_tokens: torch.Tensor,
        chemberta_tokens: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        reaction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if protein_tokens.ndim != 3:
            raise ValueError(
                "BTRIKcat expects protein input [B,L,protein_dim], "
                f"got {tuple(protein_tokens.shape)}"
            )
        expected_dim = int(self.protein_proj[0].normalized_shape[0])
        if protein_tokens.size(-1) != expected_dim:
            raise ValueError(
                f"Expected protein feature width {expected_dim}, got {protein_tokens.size(-1)}"
            )

        protein = self.protein_proj(self._ensure_3d(protein_tokens))
        rxnfp = self.rxnfp_proj(self._ensure_3d(rxnfp_tokens))
        chemberta = self.chemberta_proj(self._ensure_3d(chemberta_tokens))
        protein = protein.transpose(1, 2)
        protein = (protein + self.protein_local_conv(protein)).transpose(1, 2)

        if self.use_type_embeddings:
            protein = protein + self.protein_type
            rxnfp = rxnfp + self.rxnfp_type
            chemberta = chemberta + self.chemberta_type
        protein = protein + self._resize_pos(self.protein_pos, protein.size(1))
        chemberta = chemberta + self._resize_pos(
            self.reaction_pos[:, 1:, :], chemberta.size(1)
        )

        reaction = self._fuse_reaction_tokens(rxnfp, chemberta)
        if reaction_mask is None:
            reaction_mask = torch.ones(
                reaction.size(0), reaction.size(1), dtype=torch.bool, device=reaction.device
            )
        else:
            reaction_mask = reaction_mask.bool()[:, :reaction.size(1)].clone()
            reaction_mask[:, 0] = True

        for block in self.protein_self_in:
            protein = block(protein, key_padding_mask=protein_mask)
        for block in self.reaction_self_in:
            reaction = block(reaction, key_padding_mask=reaction_mask)
        for block in self.bi_cross_blocks:
            protein, reaction = block(protein, reaction, protein_mask, reaction_mask)

        protein_pool = self._pool_protein(protein, self.protein_pooler, protein_mask)
        cls = self.cls_norm(reaction[:, 0, :])
        reaction_tokens = reaction[:, 1:, :]
        token_mask = reaction_mask[:, 1:]
        if reaction_tokens.size(1):
            if self.protein_pooler is not None:
                attn_pool = self.protein_pooler(reaction_tokens, token_mask)
            else:
                attn_pool = reaction_tokens.mean(dim=1)
            max_pool = reaction_tokens.masked_fill(
                ~token_mask.unsqueeze(-1), torch.finfo(reaction_tokens.dtype).min
            ).max(dim=1).values
        else:
            attn_pool = max_pool = cls
        reaction_pool = self.reaction_fusion_mlp(
            torch.cat([cls, self.attn_norm(attn_pool), self.max_norm(max_pool)], dim=-1)
        )

        if self.fusion == "interaction":
            fused = torch.cat(
                [protein_pool, reaction_pool, protein_pool * reaction_pool,
                 torch.abs(protein_pool - reaction_pool)], dim=-1
            )
        elif self.fusion == "concat":
            fused = torch.cat([protein_pool, reaction_pool], dim=-1)
        elif self.fusion == "gated_concat":
            gate = self.gate(torch.cat([protein_pool, reaction_pool], dim=-1))
            fused = torch.cat([gate * protein_pool, (1.0 - gate) * reaction_pool], dim=-1)
        elif self.fusion == "gated_sum":
            gate = self.gate(torch.cat([protein_pool, reaction_pool], dim=-1))
            fused = gate * protein_pool + (1.0 - gate) * reaction_pool
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
        return self.head(fused).view(-1)

class _ProteinToReactionBlock(nn.Module):
    """Single-direction protein-query/reaction-key-value cross-attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, ffn_expansion: int):
        super().__init__()
        self.p_norm_q = nn.LayerNorm(d_model)
        self.r_norm_kv = nn.LayerNorm(d_model)
        self.p_attends_r = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.p_norm_ffn = nn.LayerNorm(d_model)
        self.p_ffn = FeedForward(d_model, dropout, ffn_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        protein: torch.Tensor,
        reaction: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        reaction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.p_norm_q(protein)
        key_value = self.r_norm_kv(reaction)
        if protein_mask is not None:
            query = query.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        if reaction_mask is not None:
            key_value = key_value.masked_fill(~reaction_mask.unsqueeze(-1), 0.0)
        delta, _ = self.p_attends_r(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=~reaction_mask if reaction_mask is not None else None,
            need_weights=False,
        )
        protein = protein + self.dropout(delta)
        protein = protein + self.p_ffn(self.p_norm_ffn(protein))
        if protein_mask is not None:
            protein = protein.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        return protein

class _ReactionToProteinBlock(nn.Module):
    """Single-direction reaction-query/protein-key-value cross-attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, ffn_expansion: int):
        super().__init__()
        self.r_norm_q = nn.LayerNorm(d_model)
        self.p_norm_kv = nn.LayerNorm(d_model)
        self.r_attends_p = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.r_norm_ffn = nn.LayerNorm(d_model)
        self.r_ffn = FeedForward(d_model, dropout, ffn_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        protein: torch.Tensor,
        reaction: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        reaction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.r_norm_q(reaction)
        key_value = self.p_norm_kv(protein)
        if reaction_mask is not None:
            query = query.masked_fill(~reaction_mask.unsqueeze(-1), 0.0)
        if protein_mask is not None:
            key_value = key_value.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        delta, _ = self.r_attends_p(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=~protein_mask if protein_mask is not None else None,
            need_weights=False,
        )
        reaction = reaction + self.dropout(delta)
        reaction = reaction + self.r_ffn(self.r_norm_ffn(reaction))
        if reaction_mask is not None:
            reaction = reaction.masked_fill(~reaction_mask.unsqueeze(-1), 0.0)
        return reaction

class BTRIKcatAblation(BTRIKcat):
    """F4 variants that change only self- or cross-attention components."""

    remove_protein_self_attention = False
    remove_reaction_self_attention = False
    cross_mode = "bi"
    modality = "both"
    early_pool = False
    remove_final_interaction_fusion = False
    remove_reaction_token_fusion = False
    remove_rxnfp = False

    def __init__(self, **kwargs) -> None:
        if kwargs.get("reaction_fusion_strategy", "attention") != "attention":
            raise ValueError("F4 ablations fix reaction_fusion_strategy='attention'")
        if not self.remove_final_interaction_fusion and kwargs.get("fusion", "interaction") not in {"interaction", "concat"}:
            raise ValueError("F4 ablation fusion must be interaction or concat")
        if self.remove_final_interaction_fusion:
            # Keep the same two pooled representations but remove the
            # product and absolute-difference interaction features.
            kwargs["fusion"] = "concat"
        kwargs["reaction_fusion_strategy"] = "attention"
        super().__init__(**kwargs)
        d_model = int(kwargs.get("d_model", 256))
        num_heads = int(kwargs.get("num_heads", 4))
        dropout = float(kwargs.get("dropout", 0.0))
        ffn_expansion = int(kwargs.get("ffn_expansion", 4))
        depth = int(kwargs.get("num_layers", 1))
        if self.cross_mode == "protein_to_reaction":
            self.bi_cross_blocks = torch.nn.ModuleList()
            self.reaction_to_protein_blocks = torch.nn.ModuleList(
                [_ReactionToProteinBlock(d_model, num_heads, dropout, ffn_expansion) for _ in range(depth)]
            )
        elif self.cross_mode == "reaction_to_protein":
            self.bi_cross_blocks = torch.nn.ModuleList()
            self.uni_cross_blocks = torch.nn.ModuleList(
                [_ProteinToReactionBlock(d_model, num_heads, dropout, ffn_expansion) for _ in range(depth)]
            )
        elif self.cross_mode == "none":
            self.bi_cross_blocks = torch.nn.ModuleList()

        if self.modality in {"protein", "reaction"}:
            hidden1 = max(d_model, d_model * 2)
            hidden2 = max(d_model // 2, hidden1 // 2)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, hidden1), nn.GELU(), nn.Dropout(dropout),
                nn.LayerNorm(hidden1), nn.Linear(hidden1, hidden2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden2, 1),
            )
        if self.remove_rxnfp:
            self.reaction_cls_without_rxnfp = nn.Parameter(
                torch.zeros(1, 1, d_model)
            )
            nn.init.normal_(self.reaction_cls_without_rxnfp, std=0.02)

    def forward(
        self, protein_tokens, rxnfp_tokens, chemberta_tokens,
        protein_mask=None, reaction_mask=None,
    ):
        if protein_tokens.ndim != 3:
            raise ValueError(f"F4 expects protein input [B,L,D], got {tuple(protein_tokens.shape)}")
        expected_dim = int(self.protein_proj[0].normalized_shape[0])
        if protein_tokens.size(-1) != expected_dim:
            raise ValueError(f"Expected protein feature width {expected_dim}, got {protein_tokens.size(-1)}")

        protein = self.protein_proj(self._ensure_3d(protein_tokens))
        chemberta = self.chemberta_proj(self._ensure_3d(chemberta_tokens))
        protein = protein.transpose(1, 2)
        protein = (protein + self.protein_local_conv(protein)).transpose(1, 2)
        if self.use_type_embeddings:
            protein = protein + self.protein_type
            if not self.remove_rxnfp:
                rxnfp = self.rxnfp_proj(self._ensure_3d(rxnfp_tokens))
                rxnfp = rxnfp + self.rxnfp_type
            chemberta = chemberta + self.chemberta_type
        protein = protein + self._resize_pos(self.protein_pos, protein.size(1))
        chemberta = chemberta + self._resize_pos(self.reaction_pos[:, 1:, :], chemberta.size(1))
        if self.remove_rxnfp:
            cls = self.reaction_cls_without_rxnfp.expand(chemberta.size(0), -1, -1)
            reaction = torch.cat([cls, chemberta], dim=1)
        elif self.remove_reaction_token_fusion:
            # Ablate F4's RXNFP-to-ChemBERTa token interaction while keeping
            # both projected reaction token streams for later pooling.
            rxnfp = self.rxnfp_proj(self._ensure_3d(rxnfp_tokens))
            if self.use_type_embeddings:
                rxnfp = rxnfp + self.rxnfp_type
            reaction = torch.cat([rxnfp, chemberta], dim=1)
        else:
            rxnfp = self.rxnfp_proj(self._ensure_3d(rxnfp_tokens))
            if self.use_type_embeddings:
                rxnfp = rxnfp + self.rxnfp_type
            reaction = self._fuse_reaction_tokens(rxnfp, chemberta)
        if reaction_mask is None:
            reaction_mask = torch.ones(reaction.size(0), reaction.size(1), dtype=torch.bool, device=reaction.device)
        else:
            reaction_mask = reaction_mask.bool()[:, :reaction.size(1)].clone()
            reaction_mask[:, 0] = True

        if self.modality != "reaction" and self.early_pool:
            protein = self._pool_protein(protein, self.protein_pooler, protein_mask).unsqueeze(1)
            protein_mask = torch.ones(
                protein.size(0), 1, dtype=torch.bool, device=protein.device
            )

        if self.modality != "reaction" and not self.remove_protein_self_attention:
            for block in self.protein_self_in:
                protein = block(protein, key_padding_mask=protein_mask)
        if self.modality != "protein" and not self.remove_reaction_self_attention:
            for block in self.reaction_self_in:
                reaction = block(reaction, key_padding_mask=reaction_mask)
        if self.modality == "both" and self.cross_mode == "bi":
            for block in self.bi_cross_blocks:
                protein, reaction = block(protein, reaction, protein_mask, reaction_mask)
        elif self.modality == "both" and self.cross_mode == "protein_to_reaction":
            for block in self.reaction_to_protein_blocks:
                reaction = block(protein, reaction, protein_mask, reaction_mask)
        elif self.modality == "both" and self.cross_mode == "reaction_to_protein":
            for block in self.uni_cross_blocks:
                protein = block(protein, reaction, protein_mask, reaction_mask)

        protein_pool = None
        reaction_pool = None
        if self.modality != "reaction":
            protein_pool = self._pool_protein(protein, self.protein_pooler, protein_mask)
        if self.modality != "protein":
            cls = self.cls_norm(reaction[:, 0, :])
            tokens, token_mask = reaction[:, 1:, :], reaction_mask[:, 1:]
            if tokens.size(1):
                attn_pool = self.protein_pooler(tokens, token_mask) if self.protein_pooler is not None else tokens.mean(1)
                max_pool = tokens.masked_fill(~token_mask.unsqueeze(-1), torch.finfo(tokens.dtype).min).max(1).values
            else:
                attn_pool = max_pool = cls
            reaction_stats = torch.cat(
                [cls, self.attn_norm(attn_pool), self.max_norm(max_pool)], dim=-1
            )
            reaction_pool = self.reaction_fusion_mlp(reaction_stats)

        if self.modality == "protein":
            return self.head(protein_pool).view(-1)
        if self.modality == "reaction":
            return self.head(reaction_pool).view(-1)
        if self.fusion == "interaction":
            fused = torch.cat([protein_pool, reaction_pool, protein_pool * reaction_pool,
                               torch.abs(protein_pool - reaction_pool)], dim=-1)
        elif self.fusion == "concat":
            fused = torch.cat([protein_pool, reaction_pool], dim=-1)
        else:
            raise ValueError(f"Unsupported F4 ablation fusion: {self.fusion}")
        return self.head(fused).view(-1)

class BTRIKcatNoProteinSelfAttention(BTRIKcatAblation):
    remove_protein_self_attention = True

class BTRIKcatNoReactionSelfAttention(BTRIKcatAblation):
    remove_reaction_self_attention = True

class BTRIKcatProteinToReactionOnly(BTRIKcatAblation):
    """Protein tokens provide context that updates reaction tokens."""

    cross_mode = "protein_to_reaction"

class BTRIKcatReactionToProteinOnly(BTRIKcatAblation):
    """Reaction tokens provide context that updates protein tokens."""

    cross_mode = "reaction_to_protein"

class BTRIKcatProteinOnly(BTRIKcatAblation):
    modality = "protein"
    cross_mode = "none"

class BTRIKcatReactionOnly(BTRIKcatAblation):
    modality = "reaction"
    cross_mode = "none"

class BTRIKcatEarlyPool(BTRIKcatAblation):
    """Pool protein early, with both self-attention branches removed."""

    early_pool = True
    remove_protein_self_attention = True
    remove_reaction_self_attention = True

class BTRIKcatNoCrossAttention(BTRIKcatAblation):
    """Retain both self-attention branches while removing cross-attention."""

    cross_mode = "none"

class BTRIKcatNoFinalInteractionFusion(BTRIKcatAblation):
    """Remove final product/difference interaction features."""

    remove_final_interaction_fusion = True

class BTRIKcatNoReactionFusion(BTRIKcatAblation):
    """Remove RXNFP-ChemBERTa token fusion, retaining pooled reaction fusion."""

    remove_reaction_token_fusion = True

class BTRIKcatWithoutRXNFP(BTRIKcatAblation):
    """Remove RXNFP and use a learned reaction CLS with ChemBERTa tokens."""

    remove_rxnfp = True

class BTRIKcatWithoutChemBERTa(BTRIKcat):
    """Use the established empty-ChemBERTa fallback of the full model."""
    def forward(self, protein_tokens, rxnfp_tokens, chemberta_tokens,
                protein_mask=None, reaction_mask=None):
        empty = chemberta_tokens[:, :0, :] if chemberta_tokens.ndim == 3 else chemberta_tokens.unsqueeze(1)[:, :0, :]
        return super().forward(protein_tokens, rxnfp_tokens, empty,
                               protein_mask=protein_mask,
                               reaction_mask=None if reaction_mask is None else reaction_mask[:, :1])

MODEL_VARIANTS = {
    "full": BTRIKcat,
    "no_protein_self_attention": BTRIKcatNoProteinSelfAttention,
    "no_reaction_self_attention": BTRIKcatNoReactionSelfAttention,
    "protein_to_reaction_only": BTRIKcatProteinToReactionOnly,
    "reaction_to_protein_only": BTRIKcatReactionToProteinOnly,
    "protein_only": BTRIKcatProteinOnly,
    "reaction_only": BTRIKcatReactionOnly,
    "early_pool": BTRIKcatEarlyPool,
    "no_cross_attention": BTRIKcatNoCrossAttention,
    "no_final_interaction_fusion": BTRIKcatNoFinalInteractionFusion,
    "no_reaction_fusion": BTRIKcatNoReactionFusion,
    "without_rxnfp": BTRIKcatWithoutRXNFP,
    "without_chemberta": BTRIKcatWithoutChemBERTa,
    "no_ranking_loss": BTRIKcat,
}

__all__ = ["BTRIKcat", "MODEL_VARIANTS"]

