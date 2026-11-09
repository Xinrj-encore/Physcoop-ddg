import torch
import torch.nn as nn
import torch.nn.functional as F

from physcoop.model import PhysCoopProteinMPNN, sequence_scores


class DDGPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if getattr(cfg, "model_arch", "physcoop_ddg") != "physcoop_ddg":
            raise ValueError("This release supports only model_arch='physcoop_ddg'")
        if getattr(cfg, "use_physquad", False):
            raise ValueError("The legacy PhysQuad branch is not part of this release")

        self.mpnn = PhysCoopProteinMPNN(
            ca_only=cfg.ca_only,
            num_letters=21,
            node_features=cfg.hidden_dim,
            edge_features=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_layers,
            num_decoder_layers=cfg.num_layers,
            augment_eps=cfg.backbone_noise,
            k_neighbors=cfg.num_edges,
            num_tri_heads=cfg.num_tri_heads,
            max_num_hyperedges=cfg.max_num_hyperedges,
            num_mut_subgraph_nodes=cfg.num_mut_subgraph_nodes,
            hyper_ratio=cfg.hyper_ratio,
            num_edges_ratio=cfg.num_edges_ratio,
            edges_selection=cfg.edges_selection,
            use_mutation_subgraph=getattr(cfg, "use_mutation_subgraph", True),
            phys_hyperedge_attr_dim=getattr(cfg, "phys_hyperedge_attr_dim", 8),
            physcoop_select_k=getattr(cfg, "physcoop_select_k", 16),
            physcoop_candidate_k=getattr(cfg, "physcoop_candidate_k", None),
            physcoop_max_k=getattr(cfg, "physcoop_max_k", None),
            physcoop_num_heads=getattr(cfg, "physcoop_num_heads", 4),
            physcoop_dropout=getattr(cfg, "physcoop_dropout", 0.05),
            physcoop_distance_scale=getattr(cfg, "physcoop_distance_scale", 10.0),
            physcoop_base_member_bias=getattr(cfg, "physcoop_base_member_bias", 4.0),
            physcoop_mutation_bias=getattr(cfg, "physcoop_mutation_bias", 1.5),
            physcoop_distance_bias=getattr(cfg, "physcoop_distance_bias", 2.0),
            physcoop_alpha_init=getattr(cfg, "physcoop_alpha_init", 0.01),
            physcoop_alpha_max=getattr(cfg, "physcoop_alpha_max", 0.25),
            physcoop_static_members=getattr(cfg, "physcoop_static_members", False),
            physcoop_adaptive_members=getattr(cfg, "physcoop_adaptive_members", False),
            physcoop_min_k=getattr(cfg, "physcoop_min_k", 4),
            physcoop_member_temperature=getattr(cfg, "physcoop_member_temperature", 0.5),
            physcoop_capacity_init=getattr(cfg, "physcoop_capacity_init", 8.0),
            physcoop_disable_cooperation=getattr(
                cfg, "physcoop_disable_cooperation", False
            ),
            physcoop_disable_pairwise_bias=getattr(
                cfg, "physcoop_disable_pairwise_bias", False
            ),
            physcoop_disable_mutation_context=getattr(
                cfg, "physcoop_disable_mutation_context", False
            ),
        )
        self.boltzmann_scalar = nn.Parameter(torch.ones(1))
        self.loss_weight_boltzmann = cfg.loss_weight_boltzmann

    @staticmethod
    def _thermodynamic_cycle(
        wt_scores,
        mut_scores,
        num_mutation_chains,
        complex_indices,
        single_chain_indices,
        device,
    ):
        chain_ids = torch.arange(len(num_mutation_chains), device=device)
        chain_ids = chain_ids.repeat_interleave(
            torch.tensor(num_mutation_chains, device=device)
        )
        wt_complex = wt_scores.index_select(0, complex_indices)
        wt_chain_sum = torch.zeros_like(wt_complex)
        wt_chain_sum.scatter_add_(
            0, chain_ids, wt_scores.index_select(0, single_chain_indices)
        )
        mut_complex = mut_scores.index_select(0, complex_indices)
        mut_chain_sum = torch.zeros_like(mut_complex)
        mut_chain_sum.scatter_add_(
            0, chain_ids, mut_scores.index_select(0, single_chain_indices)
        )
        return wt_complex - wt_chain_sum, mut_complex - mut_chain_sum

    def forward(self, batch):
        h_edges = self.mpnn.deterministic_forward(
            batch["X"],
            batch["aa"],
            batch["mask"],
            batch["chain_M"],
            batch["residue_idx"],
            batch["chain_encoding_all"],
            phys_hyperedge_index=batch["phys_hyperedge_index"],
            phys_hyperedge_node_mask=batch["phys_hyperedge_node_mask"],
            phys_hyperedge_mask=batch["phys_hyperedge_mask"],
            phys_hyperedge_attr=batch["phys_hyperedge_attr"],
            delta_sasa=batch.get("delta_sasa"),
        )
        wt_scores = sequence_scores(batch["aa"], h_edges, batch["mask"])
        mut_scores = sequence_scores(batch["aa_mut"], h_edges, batch["mask"])

        device = h_edges.device
        complex_indices = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=device),
                torch.cumsum(
                    torch.tensor(batch["num_mut_chains"], device=device), dim=0
                ),
            )
        )[:-1] + torch.arange(len(batch["num_mut_chains"]), device=device)
        all_indices = torch.arange(h_edges.shape[0], device=device)
        single_chain_indices = all_indices[~torch.isin(all_indices, complex_indices)]

        wt_cycle, mut_cycle = self._thermodynamic_cycle(
            wt_scores,
            mut_scores,
            batch["num_mut_chains"],
            complex_indices,
            single_chain_indices,
            device,
        )
        ddg_pred = (mut_cycle - wt_cycle) * self.boltzmann_scalar
        ddg_true = batch["ddG"][complex_indices]
        loss = F.mse_loss(ddg_pred, ddg_true) * self.loss_weight_boltzmann
        return loss, {"ddG_pred": ddg_pred, "ddG_true": ddg_true}, h_edges
