import unittest
from types import SimpleNamespace

import torch

from ddg_predictor import DDGPredictor


class DDGPredictorTest(unittest.TestCase):
    def test_forward_backward_with_thermodynamic_cycle(self):
        config = SimpleNamespace(
            model_arch="physcoop_ddg",
            ca_only=False,
            hidden_dim=32,
            num_layers=2,
            backbone_noise=0.0,
            num_edges=8,
            num_tri_heads=4,
            use_mutation_subgraph=True,
            max_num_hyperedges=2,
            num_mut_subgraph_nodes=4,
            hyper_ratio=2,
            num_edges_ratio=2,
            edges_selection="distance_min",
            phys_hyperedge_attr_dim=8,
            physcoop_select_k=4,
            physcoop_num_heads=4,
            physcoop_dropout=0.0,
            physcoop_static_members=False,
            physcoop_adaptive_members=True,
            physcoop_min_k=2,
            physcoop_member_temperature=0.5,
            physcoop_capacity_init=3,
            physcoop_disable_cooperation=False,
            physcoop_disable_pairwise_bias=False,
            physcoop_disable_mutation_context=False,
            use_physquad=False,
            loss_weight_boltzmann=1.0,
        )
        model = DDGPredictor(config)
        batch_size, length, num_edges = 2, 16, 2
        coordinates = torch.randn(batch_size, length, 4, 3)
        sequence = torch.randint(0, 21, (batch_size, length))
        mutant = sequence.clone()
        mutant[:, 3] = (mutant[:, 3] + 1) % 21
        mutation_mask = torch.zeros(batch_size, length)
        mutation_mask[:, 3] = 1
        chain_encoding = torch.zeros(batch_size, length, dtype=torch.long)
        chain_encoding[0, :8] = 1
        chain_encoding[0, 8:] = 2
        chain_encoding[1, :8] = 1
        hyperedge_index = torch.tensor(
            [[[0, 1, 8, 9], [5, 6, 13, 14]], [[0, 1, 2, 3], [0, 0, 0, 0]]]
        )
        hyperedge_node_mask = torch.tensor(
            [[[1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [0, 0, 0, 0]]],
            dtype=torch.bool,
        )
        batch = {
            "X": coordinates,
            "aa": sequence,
            "aa_mut": mutant,
            "mask": torch.ones(batch_size, length),
            "chain_M": mutation_mask,
            "chain_encoding_all": chain_encoding,
            "residue_idx": torch.arange(1, length + 1).repeat(batch_size, 1),
            "phys_hyperedge_index": hyperedge_index,
            "phys_hyperedge_node_mask": hyperedge_node_mask,
            "phys_hyperedge_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            "phys_hyperedge_attr": torch.rand(batch_size, num_edges, 8),
            "delta_sasa": torch.zeros(batch_size, length),
            "ddG": torch.tensor([0.5]),
            "num_mut_chains": [1],
            "complex": ["example", "example"],
            "num_muts": [1],
            "id": [0],
        }

        loss, output, log_probs = model(batch)
        self.assertEqual(log_probs.shape, (batch_size, length, 21))
        self.assertEqual(output["ddG_pred"].shape, (1,))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.boltzmann_scalar.grad)


if __name__ == "__main__":
    unittest.main()
