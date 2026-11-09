import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from protein_mpnn_utils import (
    ProteinMPNN,
    _scores,
    cat_neighbors_nodes,
    gather_nodes,
)
from utils import ScaleAwareFusion, get_edges_in_cluster

from .layers import (
    PhysicalGuidedAdaptiveLearnedHypergraph,
    physical_hyperedge_to_full_graph,
)


def _checkpoint(function, *args):
    try:
        return checkpoint(function, *args, use_reentrant=False)
    except TypeError:
        return checkpoint(function, *args)


class PhysCoopProteinMPNN(ProteinMPNN):
    def __init__(
        self,
        num_letters,
        node_features,
        edge_features,
        hidden_dim,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=64,
        augment_eps=0.05,
        dropout=0.1,
        ca_only=False,
        num_tri_heads=4,
        max_num_hyperedges=20,
        num_mut_subgraph_nodes=20,
        hyper_ratio=4,
        num_edges_ratio=2.0,
        edges_selection="distance_min",
        use_mutation_subgraph=True,
        phys_hyperedge_attr_dim=8,
        physcoop_select_k=16,
        physcoop_candidate_k=None,
        physcoop_max_k=None,
        physcoop_num_heads=4,
        physcoop_dropout=0.05,
        physcoop_distance_scale=10.0,
        physcoop_base_member_bias=4.0,
        physcoop_mutation_bias=1.5,
        physcoop_distance_bias=2.0,
        physcoop_alpha_init=0.01,
        physcoop_alpha_max=0.25,
        physcoop_static_members=False,
        physcoop_adaptive_members=False,
        physcoop_min_k=4,
        physcoop_member_temperature=0.5,
        physcoop_capacity_init=8.0,
        physcoop_disable_cooperation=False,
        physcoop_disable_pairwise_bias=False,
        physcoop_disable_mutation_context=False,
    ):
        super().__init__(
            num_letters=num_letters,
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            vocab=vocab,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
            dropout=dropout,
            ca_only=ca_only,
            num_tri_heads=num_tri_heads,
            use_hypergraph=use_mutation_subgraph,
            max_num_hyperedges=max_num_hyperedges,
            num_mut_subgraph_nodes=num_mut_subgraph_nodes,
            hyper_ratio=hyper_ratio,
            num_edges_ratio=num_edges_ratio,
            edges_selection=edges_selection,
        )
        if hasattr(self, "TripletLayers"):
            del self.TripletLayers

        self.use_mutation_subgraph = bool(use_mutation_subgraph)
        self.PhysicalGuidedAdaptiveHypergraphLayers = nn.ModuleList(
            [
                PhysicalGuidedAdaptiveLearnedHypergraph(
                    hidden_dim,
                    attr_dim=phys_hyperedge_attr_dim,
                    select_k=physcoop_select_k,
                    candidate_k=physcoop_candidate_k,
                    max_k=physcoop_max_k,
                    num_heads=physcoop_num_heads,
                    dropout=physcoop_dropout,
                    distance_scale=physcoop_distance_scale,
                    base_member_bias=physcoop_base_member_bias,
                    mutation_bias=physcoop_mutation_bias,
                    distance_bias=physcoop_distance_bias,
                    alpha_init=physcoop_alpha_init,
                    alpha_max=physcoop_alpha_max,
                    static_members=physcoop_static_members,
                    adaptive_members=physcoop_adaptive_members,
                    min_k=physcoop_min_k,
                    member_temperature=physcoop_member_temperature,
                    capacity_init=physcoop_capacity_init,
                    use_cooperative_attention=not physcoop_disable_cooperation,
                    use_pairwise_physical_bias=not physcoop_disable_pairwise_bias,
                    use_mutation_context=not physcoop_disable_mutation_context,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.physical_attr_mlp = nn.Sequential(
            nn.Linear(phys_hyperedge_attr_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.physical_fusion = ScaleAwareFusion(
            hidden_dim,
            alpha_init=0.001,
            alpha_max=0.1,
            dropout=0.1,
        )

    def _apply_mutation_subgraph(
        self,
        layer_index,
        h_V,
        ca_coords,
        mutation_mask,
        mask,
    ):
        if not self.use_mutation_subgraph:
            return h_V

        neighborhood_h = []
        neighborhood_ca = []
        neighborhood_indices = []
        batch_size, length, _ = h_V.shape

        for batch_id in range(batch_size):
            valid_indices = mask[batch_id].nonzero(as_tuple=True)[0]
            mutated_indices = mutation_mask[batch_id].nonzero(as_tuple=True)[0]
            if valid_indices.numel() == 0 or mutated_indices.numel() == 0:
                continue

            valid_ca = ca_coords[batch_id, valid_indices]
            for mutation_index in mutated_indices:
                if not mask[batch_id, mutation_index]:
                    continue
                distances = torch.cdist(
                    ca_coords[batch_id, mutation_index].unsqueeze(0), valid_ca
                ).squeeze(0)
                distances[valid_indices == mutation_index] = torch.finfo(distances.dtype).max
                neighbor_count = min(
                    self.num_mut_subgraph_nodes - 1, valid_indices.numel() - 1
                )
                if neighbor_count <= 0:
                    continue
                _, neighbor_positions = torch.topk(
                    distances, k=neighbor_count, largest=False
                )
                indices = torch.cat(
                    [mutation_index.unsqueeze(0), valid_indices[neighbor_positions]]
                )
                neighborhood_h.append(h_V[batch_id, indices])
                neighborhood_ca.append(ca_coords[batch_id, indices])
                neighborhood_indices.append((batch_id, indices))

        if not neighborhood_h:
            return h_V

        lengths = [item.size(0) for item in neighborhood_h]
        max_length = max(lengths)
        h_nodes = torch.zeros(
            len(neighborhood_h), max_length, self.hidden_dim,
            device=h_V.device, dtype=h_V.dtype,
        )
        ca_nodes = torch.zeros(
            len(neighborhood_h), max_length, 3,
            device=ca_coords.device, dtype=ca_coords.dtype,
        )
        node_mask = torch.zeros(
            len(neighborhood_h), max_length,
            device=h_V.device, dtype=torch.bool,
        )
        for item_id, (h_item, ca_item, item_length) in enumerate(
            zip(neighborhood_h, neighborhood_ca, lengths)
        ):
            h_nodes[item_id, :item_length] = h_item
            ca_nodes[item_id, :item_length] = ca_item
            node_mask[item_id, :item_length] = True

        h_edges, edge_mask = get_edges_in_cluster(
            h_nodes,
            ca_nodes,
            node_mask,
            self.num_edges_ratio,
            self.cluster_edge_mlp,
            self.edges_selection,
        )
        if h_edges.numel() == 0 or edge_mask.numel() == 0 or h_edges.shape[1] == 0:
            return h_V

        node_edge_mask = node_mask.unsqueeze(-1) & edge_mask.unsqueeze(-2)
        attention_mask = (1 - node_edge_mask.unsqueeze(-1).to(h_V.dtype)) * torch.finfo(
            h_V.dtype
        ).min
        node_node = torch.einsum("bid,bjd->bijd", h_nodes, h_nodes)
        node_edge = torch.einsum("bjd,bkd->bjkd", h_nodes, h_edges)
        node_update = self.QuadrupletLayers[layer_index](
            node_node, node_edge, attention_mask
        ).mean(dim=2)

        delta = torch.zeros_like(h_V)
        for item_id, (batch_id, indices) in enumerate(neighborhood_indices):
            item_length = lengths[item_id]
            delta[batch_id, indices] = (
                node_update[item_id, :item_length] - h_nodes[item_id, :item_length]
            )
        return self.mut_fusion(h_V, delta)

    def deterministic_forward(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        phys_hyperedge_index=None,
        phys_hyperedge_node_mask=None,
        phys_hyperedge_mask=None,
        phys_hyperedge_attr=None,
        delta_sasa=None,
        foldx_features=None,
        return_aux=False,
    ):
        del delta_sasa, foldx_features
        device = X.device
        ca_coords = X[:, :, 1, :]
        E, edge_index = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros(
            (E.shape[0], E.shape[1], E.shape[-1]), device=device, dtype=E.dtype
        )
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), edge_index).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        chain_M = chain_M * mask

        for layer_index, encoder_layer in enumerate(self.encoder_layers):
            h_V, h_E = _checkpoint(
                encoder_layer, h_V, h_E, edge_index, mask, mask_attend
            )
            h_edges, edge_mask, valid_nodes, selected_idx, _ = (
                self.PhysicalGuidedAdaptiveHypergraphLayers[layer_index](
                    h_V,
                    ca_coords,
                    mask,
                    chain_M,
                    phys_hyperedge_index,
                    phys_hyperedge_node_mask,
                    phys_hyperedge_mask,
                    phys_hyperedge_attr,
                    chain_encoding=chain_encoding_all,
                )
            )
            residue_update = physical_hyperedge_to_full_graph(
                h_edges,
                edge_mask,
                valid_nodes,
                selected_idx,
                h_V.shape[1],
            )
            h_V = self.physical_fusion(h_V, residue_update)
            h_V = self._apply_mutation_subgraph(
                layer_index, h_V, ca_coords, chain_M, mask
            )

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, edge_index)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, edge_index)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, edge_index)
        decoding_order = torch.argsort(
            (chain_M + 0.0001) * torch.abs(torch.randn(chain_M.shape, device=device))
        )
        mask_size = edge_index.shape[1]
        permutation = F.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask = torch.einsum(
            "ij,biq,bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation,
            permutation,
        )
        mask_attend = torch.gather(order_mask, 2, edge_index).unsqueeze(-1)
        mask_1d = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw = mask_1d * mask_attend
        mask_fw = mask_1d * (1 - mask_attend)
        encoded = mask_fw * h_EXV_encoder
        for decoder_layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, edge_index)
            h_ESV = mask_bw * h_ESV + encoded
            h_V = _checkpoint(decoder_layer, h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        return F.log_softmax(logits, dim=-1)


def sequence_scores(sequence, log_probs, mask):
    return _scores(sequence, log_probs, mask)
