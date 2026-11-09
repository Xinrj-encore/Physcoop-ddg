import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAwareFusion(nn.Module):
    def __init__(self, dim, alpha_init=0.01, alpha_max=0.1, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.dropout = nn.Dropout(dropout)

    def forward(self, residues, update):
        return residues + self.dropout(self.alpha * self.norm(update))


def masked_mean(x, mask, dim=1, eps=1e-6):
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.to(dtype=x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    numerator = (x * mask).sum(dim=dim)
    denominator = mask.sum(dim=dim).clamp_min(eps)
    while denominator.dim() < numerator.dim():
        denominator = denominator.unsqueeze(-1)
    return numerator / denominator


def build_valid_hyperedge_masks(
    hyperedge_index,
    hyperedge_node_mask,
    hyperedge_mask,
    residue_mask,
    chain_encoding=None,
):
    batch_size, length = residue_mask.shape
    _, num_edges, max_nodes = hyperedge_index.shape
    device = residue_mask.device

    idx = torch.clamp(hyperedge_index.to(device), min=0, max=max(length - 1, 0))
    flat_idx = idx.reshape(batch_size, num_edges * max_nodes)
    node_mask = hyperedge_node_mask.to(device).bool()
    edge_mask = hyperedge_mask.to(device).bool()
    residue_valid = torch.gather(residue_mask.bool(), 1, flat_idx).reshape(
        batch_size, num_edges, max_nodes
    )
    valid_nodes = node_mask & residue_valid & edge_mask.unsqueeze(-1)
    valid_edges = edge_mask & (valid_nodes.sum(dim=-1) >= 2)

    if chain_encoding is not None:
        chain_ids = torch.gather(chain_encoding.to(device), 1, flat_idx).reshape(
            batch_size, num_edges, max_nodes
        )
        multi_chain = torch.zeros(
            batch_size, num_edges, dtype=torch.bool, device=device
        )
        for batch_id in range(batch_size):
            for edge_id in range(num_edges):
                chains = torch.unique(
                    chain_ids[batch_id, edge_id][valid_nodes[batch_id, edge_id]]
                )
                multi_chain[batch_id, edge_id] = chains.numel() >= 2
        valid_edges = valid_edges & multi_chain
        valid_nodes = valid_nodes & valid_edges.unsqueeze(-1)

    return idx, valid_edges, valid_nodes


class PhysicalInterfaceCooperativeAttention(nn.Module):
    def __init__(
        self,
        dim,
        attr_dim,
        num_heads=4,
        dropout=0.05,
        distance_scale=10.0,
        alpha_init=0.01,
        alpha_max=0.25,
        use_pairwise_physical_bias=True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if alpha_max <= 0:
            raise ValueError("alpha_max must be positive")

        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.distance_scale = float(distance_scale)
        self.alpha_max = float(alpha_max)
        self.use_pairwise_physical_bias = bool(use_pairwise_physical_bias)
        init_ratio = min(max(alpha_init / alpha_max, 1e-4), 1 - 1e-4)
        self.alpha_logit = nn.Parameter(
            torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        )

        self.edge_norm = nn.LayerNorm(dim)
        self.attr_norm = nn.LayerNorm(attr_dim)
        self.attr_proj = nn.Linear(attr_dim, dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.pair_bias = nn.Sequential(
            nn.Linear(6, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, self.num_heads),
        )
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(3 * dim + 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _masked_centers(ca_coords, idx, valid_nodes):
        batch_size, num_edges, max_nodes = idx.shape
        gathered = torch.gather(
            ca_coords,
            1,
            idx.reshape(batch_size, num_edges * max_nodes)
            .unsqueeze(-1)
            .expand(-1, -1, ca_coords.shape[-1]),
        ).reshape(batch_size, num_edges, max_nodes, ca_coords.shape[-1])
        weights = valid_nodes.to(ca_coords.dtype)
        centers = (gathered * weights.unsqueeze(-1)).sum(dim=2)
        return centers / weights.sum(dim=2, keepdim=True).clamp_min(1.0)

    def _pair_features(
        self,
        edge_mask,
        valid_nodes,
        hyperedge_index,
        ca_coords,
        mutation_mask,
        attr_context,
    ):
        batch_size, num_edges, max_nodes = hyperedge_index.shape
        device = edge_mask.device
        dtype = attr_context.dtype
        length = ca_coords.shape[1]
        idx = torch.clamp(
            hyperedge_index.to(device), min=0, max=max(length - 1, 0)
        )
        node_valid = valid_nodes.bool() & edge_mask.unsqueeze(-1)
        edge_size = node_valid.to(dtype).sum(dim=-1).clamp_min(1.0)

        membership = torch.zeros(
            batch_size, num_edges, length, device=device, dtype=dtype
        )
        membership.scatter_add_(2, idx, node_valid.to(dtype))
        membership = torch.clamp(membership, max=1.0)
        overlap = torch.einsum("bhl,bml->bhm", membership, membership)
        overlap = overlap / torch.minimum(
            edge_size.unsqueeze(2), edge_size.unsqueeze(1)
        ).clamp_min(1.0)

        centers = self._masked_centers(ca_coords, idx, node_valid)
        spatial = torch.exp(
            -torch.cdist(centers, centers) / self.distance_scale
        ).to(dtype)

        attr_unit = F.normalize(attr_context, dim=-1, eps=1e-6)
        attr_similarity = torch.einsum("bhd,bmd->bhm", attr_unit, attr_unit)
        attr_similarity = (attr_similarity + 1.0) * 0.5

        flat_idx = idx.reshape(batch_size, num_edges * max_nodes)
        mut_nodes = torch.gather(mutation_mask.bool(), 1, flat_idx).reshape(
            batch_size, num_edges, max_nodes
        )
        active = ((mut_nodes & node_valid).any(dim=-1) & edge_mask).to(dtype)
        pair_features = torch.stack(
            [
                overlap,
                spatial,
                attr_similarity,
                active.unsqueeze(2).expand(-1, -1, num_edges),
                active.unsqueeze(1).expand(-1, num_edges, -1),
                active.unsqueeze(2) * active.unsqueeze(1),
            ],
            dim=-1,
        )
        mutation_density = (mut_nodes & node_valid).to(dtype).sum(dim=-1) / edge_size
        edge_stats = torch.stack(
            [edge_size / max(float(max_nodes), 1.0), mutation_density, active],
            dim=-1,
        )
        edge_stats = torch.cat([edge_stats, edge_mask.to(dtype).unsqueeze(-1)], dim=-1)
        return pair_features, edge_stats

    def forward(
        self,
        h_edge,
        edge_mask,
        edge_attr,
        valid_nodes,
        hyperedge_index,
        ca_coords,
        mutation_mask,
    ):
        batch_size, num_edges, dim = h_edge.shape
        device = h_edge.device
        dtype = h_edge.dtype
        if num_edges == 0:
            return h_edge, torch.zeros(batch_size, dim, device=device, dtype=dtype)

        edge_mask = edge_mask.to(device).bool()
        edge_attr = edge_attr.to(device=device, dtype=dtype)
        attr_context = self.attr_proj(self.attr_norm(edge_attr))
        h_input = self.edge_norm(h_edge) + attr_context
        pair_features, edge_stats = self._pair_features(
            edge_mask,
            valid_nodes,
            hyperedge_index,
            ca_coords,
            mutation_mask,
            attr_context,
        )
        if self.use_pairwise_physical_bias:
            pair_bias = self.pair_bias(pair_features).permute(0, 3, 1, 2)
        else:
            pair_bias = torch.zeros(
                batch_size, self.num_heads, num_edges, num_edges,
                device=device, dtype=dtype,
            )

        q, k, v = self.qkv(h_input).chunk(3, dim=-1)
        q = q.reshape(batch_size, num_edges, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, num_edges, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, num_edges, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + pair_bias
        query_mask = edge_mask.unsqueeze(1).unsqueeze(-1)
        key_mask = edge_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~(query_mask & key_mask), -1e4)

        weights = torch.softmax(scores, dim=-1) * key_mask.to(dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        update = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch_size, num_edges, dim
        )
        update = self.out_proj(update)

        gate_input = torch.cat(
            [self.edge_norm(h_edge), update, attr_context, edge_stats.to(dtype)],
            dim=-1,
        )
        gate = self.gate(gate_input)
        alpha = self.alpha_max * torch.sigmoid(self.alpha_logit).to(dtype)
        h_edge = h_edge + self.dropout(alpha * gate * update)
        h_edge = h_edge + self.dropout(alpha * self.ffn(h_edge))
        h_edge = h_edge * edge_mask.unsqueeze(-1).to(dtype)
        return h_edge, masked_mean(h_edge, edge_mask, dim=1)


class PhysicalGuidedAdaptiveLearnedHypergraph(nn.Module):
    def __init__(
        self,
        dim,
        attr_dim,
        select_k=32,
        candidate_k=None,
        max_k=None,
        num_heads=4,
        dropout=0.05,
        distance_scale=10.0,
        base_member_bias=4.0,
        mutation_bias=1.5,
        distance_bias=2.0,
        alpha_init=0.01,
        alpha_max=0.25,
        static_members=False,
        adaptive_members=False,
        min_k=4,
        member_temperature=0.5,
        capacity_init=8.0,
        use_cooperative_attention=True,
        use_pairwise_physical_bias=True,
        use_mutation_context=True,
    ):
        super().__init__()
        self.candidate_k = int(select_k if candidate_k is None else candidate_k)
        self.max_k = int(select_k if max_k is None else max_k)
        self.distance_scale = float(distance_scale)
        self.base_member_bias = float(base_member_bias)
        self.mutation_bias = float(mutation_bias)
        self.distance_bias = float(distance_bias)
        self.static_members = bool(static_members)
        self.adaptive_members = bool(adaptive_members)
        self.min_k = int(min_k)
        self.member_temperature = float(member_temperature)
        self.capacity_init = float(capacity_init)
        self.use_cooperative_attention = bool(use_cooperative_attention)
        self.use_mutation_context = bool(use_mutation_context)
        self.trace_enabled = False
        self.last_trace = None

        self.node_norm = nn.LayerNorm(dim)
        self.anchor_norm = nn.LayerNorm(dim)
        self.attr_norm = nn.LayerNorm(attr_dim)
        self.attr_proj = nn.Linear(attr_dim, dim)
        self.score = nn.Sequential(
            nn.Linear(dim + 5, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        if self.adaptive_members:
            self.member_capacity = nn.Sequential(
                nn.LayerNorm(2 * dim + attr_dim + 3),
                nn.Linear(2 * dim + attr_dim + 3, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, 1),
            )
        self.edge_gate = nn.Sequential(
            nn.LayerNorm(2 * dim + attr_dim + 4),
            nn.Linear(2 * dim + attr_dim + 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.cooperative_attention = PhysicalInterfaceCooperativeAttention(
            dim,
            attr_dim=attr_dim,
            num_heads=num_heads,
            dropout=dropout,
            distance_scale=distance_scale,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            use_pairwise_physical_bias=use_pairwise_physical_bias,
        )

        nn.init.normal_(self.score[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.score[-1].bias)
        if self.adaptive_members:
            nn.init.zeros_(self.member_capacity[-1].weight)
            nn.init.zeros_(self.member_capacity[-1].bias)
            capacity_min = min(max(self.min_k, 1), max(self.candidate_k, 1))
            capacity_max = min(max(self.max_k, capacity_min), max(self.candidate_k, 1))
            if capacity_max > capacity_min:
                init_capacity = min(
                    max(self.capacity_init, capacity_min + 1e-4),
                    capacity_max - 1e-4,
                )
                probability = (init_capacity - capacity_min) / (
                    capacity_max - capacity_min
                )
                with torch.no_grad():
                    self.member_capacity[-1].bias.fill_(
                        math.log(probability / (1.0 - probability))
                    )

    @staticmethod
    def _gather_nodes(h_V, idx):
        batch_size, num_edges, max_nodes = idx.shape
        dim = h_V.shape[-1]
        return torch.gather(
            h_V,
            1,
            idx.reshape(batch_size, num_edges * max_nodes)
            .unsqueeze(-1)
            .expand(-1, -1, dim),
        ).reshape(batch_size, num_edges, max_nodes, dim)

    @staticmethod
    def _nearest_mutation_features(ca_coords, residue_mask, mutation_mask):
        batch_size, length = residue_mask.shape
        device = ca_coords.device
        dtype = ca_coords.dtype
        mutation_mask = mutation_mask.bool() & residue_mask.bool()
        has_mutation = mutation_mask.any(dim=1)
        pair_dist = torch.cdist(ca_coords, ca_coords)
        nearest_dist = pair_dist.masked_fill(
            ~mutation_mask.unsqueeze(1), 1e4
        ).min(dim=-1).values
        nearest_dist = torch.where(
            has_mutation.unsqueeze(-1),
            nearest_dist,
            torch.zeros(batch_size, length, device=device, dtype=dtype),
        )
        return torch.exp(-nearest_dist / 8.0), torch.clamp(nearest_dist, max=32.0) / 32.0

    @staticmethod
    def _same_mutation_chain(residue_mask, mutation_mask, chain_encoding):
        if chain_encoding is None:
            return torch.zeros_like(residue_mask, dtype=torch.bool)
        mutation_mask = mutation_mask.bool() & residue_mask.bool()
        return (
            (chain_encoding.unsqueeze(2) == chain_encoding.unsqueeze(1))
            & mutation_mask.unsqueeze(1)
        ).any(dim=2)

    def forward(
        self,
        h_V,
        ca_coords,
        residue_mask,
        mutation_mask,
        phys_hyperedge_index,
        phys_hyperedge_node_mask,
        phys_hyperedge_mask,
        edge_attr,
        chain_encoding=None,
    ):
        batch_size, length, dim = h_V.shape
        _, num_edges, _ = phys_hyperedge_index.shape
        device = h_V.device
        dtype = h_V.dtype

        idx, anchor_mask, anchor_nodes_mask = build_valid_hyperedge_masks(
            phys_hyperedge_index,
            phys_hyperedge_node_mask,
            phys_hyperedge_mask,
            residue_mask,
            chain_encoding=chain_encoding,
        )
        edge_attr = edge_attr.to(device=device, dtype=dtype)
        residue_valid = residue_mask.bool()
        mutation_mask = mutation_mask.bool() & residue_valid
        if not self.use_mutation_context:
            mutation_mask = torch.zeros_like(mutation_mask)

        anchor_nodes = self._gather_nodes(h_V, idx)
        anchor_weights = anchor_nodes_mask.to(dtype)
        anchor_denom = anchor_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        anchor_embed = (anchor_nodes * anchor_weights.unsqueeze(-1)).sum(dim=2)
        anchor_embed = anchor_embed / anchor_denom
        attr_context = self.attr_proj(self.attr_norm(edge_attr))
        anchor_embed = (anchor_embed + attr_context) * anchor_mask.unsqueeze(-1).to(dtype)

        anchor_ca = self._gather_nodes(ca_coords, idx)
        anchor_centers = (
            anchor_ca * anchor_weights.unsqueeze(-1).to(ca_coords.dtype)
        ).sum(dim=2)
        anchor_centers = anchor_centers / anchor_weights.sum(
            dim=-1, keepdim=True
        ).to(ca_coords.dtype).clamp_min(1.0)
        dist_to_anchor = torch.linalg.norm(
            ca_coords.unsqueeze(1) - anchor_centers.unsqueeze(2), dim=-1
        )
        spatial_prior = torch.exp(-dist_to_anchor / self.distance_scale).to(dtype)

        base_membership = torch.zeros(
            batch_size, num_edges, length, device=device, dtype=dtype
        )
        base_membership.scatter_add_(2, idx, anchor_nodes_mask.to(dtype))
        base_membership = torch.clamp(base_membership, max=1.0)

        near_mutation, mutation_distance = self._nearest_mutation_features(
            ca_coords, residue_valid, mutation_mask
        )
        same_mut_chain = self._same_mutation_chain(
            residue_valid, mutation_mask, chain_encoding
        )

        node_context = self.node_norm(h_V).unsqueeze(1).expand(-1, num_edges, -1, -1)
        anchor_context = self.anchor_norm(anchor_embed).unsqueeze(2)
        score_features = torch.stack(
            [
                base_membership,
                spatial_prior,
                mutation_mask.unsqueeze(1).expand(-1, num_edges, -1).to(dtype),
                near_mutation.unsqueeze(1).expand(-1, num_edges, -1).to(dtype),
                same_mut_chain.unsqueeze(1).expand(-1, num_edges, -1).to(dtype),
            ],
            dim=-1,
        )
        logits = self.score(
            torch.cat(
                [node_context + anchor_context + attr_context.unsqueeze(2), score_features],
                dim=-1,
            )
        ).squeeze(-1)
        logits = logits + self.base_member_bias * base_membership
        logits = logits + self.distance_bias * spatial_prior
        logits = logits + self.mutation_bias * mutation_mask.unsqueeze(1).to(dtype)
        logits = logits + 0.5 * self.mutation_bias * near_mutation.unsqueeze(1).to(dtype)
        logits = logits + 0.25 * self.mutation_bias * same_mut_chain.unsqueeze(1).to(dtype)
        logits = logits.masked_fill(~residue_valid.unsqueeze(1), -1e4)
        logits = logits.masked_fill(~anchor_mask.unsqueeze(-1), -1e4)

        if self.static_members:
            selected_idx = idx
            selected_valid = anchor_nodes_mask & anchor_mask.unsqueeze(-1)
            selected_nodes = anchor_nodes
            selected_weights = selected_valid.to(dtype)
        else:
            select_k = min(self.candidate_k, length)
            selected_idx = torch.topk(logits, k=select_k, dim=-1).indices
            candidate_valid = torch.gather(
                residue_valid.unsqueeze(1).expand(-1, num_edges, -1),
                2,
                selected_idx,
            ) & anchor_mask.unsqueeze(-1)
            selected_logits = torch.gather(logits, 2, selected_idx)
            selected_logits = selected_logits.masked_fill(~candidate_valid, -1e4)
            selected_nodes = torch.gather(
                h_V.unsqueeze(1).expand(-1, num_edges, -1, -1),
                2,
                selected_idx.unsqueeze(-1).expand(-1, -1, -1, dim),
            )

            if self.adaptive_members:
                anchor_mutation = torch.gather(
                    mutation_mask.unsqueeze(1).expand(-1, num_edges, -1), 2, idx
                ) & anchor_nodes_mask
                anchor_near_mutation = torch.gather(
                    near_mutation.unsqueeze(1).expand(-1, num_edges, -1), 2, idx
                )
                anchor_count = anchor_nodes_mask.to(dtype).sum(dim=-1).clamp_min(1.0)
                capacity_features = torch.cat(
                    [
                        self.anchor_norm(anchor_embed),
                        attr_context,
                        edge_attr,
                        torch.stack(
                            [
                                anchor_nodes_mask.to(dtype).mean(dim=-1),
                                anchor_mutation.to(dtype).sum(dim=-1) / anchor_count,
                                (
                                    anchor_near_mutation
                                    * anchor_nodes_mask.to(dtype)
                                ).sum(dim=-1)
                                / anchor_count,
                            ],
                            dim=-1,
                        ),
                    ],
                    dim=-1,
                )
                effective_capacity = self.min_k + (
                    min(max(self.max_k, self.min_k), select_k) - self.min_k
                ) * torch.sigmoid(self.member_capacity(capacity_features).squeeze(-1))
                rank = torch.arange(select_k, device=device, dtype=dtype).view(1, 1, -1)
                keep_prob = torch.sigmoid(
                    (effective_capacity.unsqueeze(-1) - rank - 0.5)
                    / max(self.member_temperature, 1e-3)
                )
                hard_k = torch.round(effective_capacity).long().clamp(
                    min=self.min_k,
                    max=min(self.max_k, select_k),
                )
                selected_valid = candidate_valid & (rank.long() < hard_k.unsqueeze(-1))
                keep_gate = selected_valid.to(dtype) + keep_prob - keep_prob.detach()
            else:
                selected_valid = candidate_valid
                keep_gate = selected_valid.to(dtype)

            selected_weights = torch.softmax(selected_logits, dim=-1) * keep_gate
            selected_weights = selected_weights * candidate_valid.to(dtype)

        selected_weights = selected_weights / selected_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        learned_edge = (selected_nodes * selected_weights.unsqueeze(-1)).sum(dim=2)
        selected_base = torch.gather(base_membership, 2, selected_idx)
        selected_near_mut = torch.gather(
            near_mutation.unsqueeze(1).expand(-1, num_edges, -1), 2, selected_idx
        ).to(dtype)
        selected_mut_dist = torch.gather(
            mutation_distance.unsqueeze(1).expand(-1, num_edges, -1), 2, selected_idx
        ).to(dtype)
        selected_stats = torch.stack(
            [
                selected_valid.to(dtype).mean(dim=-1),
                (selected_base * selected_weights).sum(dim=-1),
                (selected_near_mut * selected_weights).sum(dim=-1),
                (selected_mut_dist * selected_weights).sum(dim=-1),
            ],
            dim=-1,
        )
        recruit_gate = self.edge_gate(
            torch.cat([anchor_embed, learned_edge, edge_attr, selected_stats], dim=-1)
        )
        learned_edge = anchor_embed + recruit_gate * (learned_edge - anchor_embed)
        learned_edge = learned_edge * anchor_mask.unsqueeze(-1).to(dtype)

        if self.use_cooperative_attention:
            learned_edge, context = self.cooperative_attention(
                learned_edge,
                anchor_mask,
                edge_attr,
                selected_valid,
                selected_idx,
                ca_coords,
                mutation_mask,
            )
        else:
            context = masked_mean(learned_edge, anchor_mask, dim=1)

        if self.trace_enabled:
            self.last_trace = {
                "selected_idx": selected_idx.detach(),
                "selected_valid": selected_valid.detach(),
                "selected_weights": selected_weights.detach(),
                "selected_count": selected_valid.sum(dim=-1).detach(),
                "anchor_mask": anchor_mask.detach(),
                "base_membership": base_membership.detach(),
                "logits": logits.detach(),
            }
        return learned_edge, anchor_mask, selected_valid, selected_idx, context


def physical_hyperedge_to_full_graph(
    h_edge,
    valid_edge_mask,
    valid_node_mask,
    hyperedge_index,
    length,
):
    batch_size, num_edges, dim = h_edge.shape
    device = h_edge.device
    output = torch.zeros(batch_size, length, dim, device=device, dtype=h_edge.dtype)
    counts = torch.zeros(batch_size, length, 1, device=device, dtype=h_edge.dtype)
    idx = torch.clamp(hyperedge_index.to(device), min=0, max=max(length - 1, 0))

    for batch_id in range(batch_size):
        for edge_id in range(num_edges):
            if not bool(valid_edge_mask[batch_id, edge_id]):
                continue
            nodes = idx[batch_id, edge_id][valid_node_mask[batch_id, edge_id]]
            if nodes.numel() == 0:
                continue
            update = h_edge[batch_id, edge_id].unsqueeze(0).expand(nodes.numel(), dim)
            output[batch_id].index_add_(0, nodes, update)
            counts[batch_id].index_add_(
                0, nodes, torch.ones(nodes.numel(), 1, device=device, dtype=h_edge.dtype)
            )

    return output / torch.clamp(counts, min=1.0)
