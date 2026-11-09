import unittest

import torch

from physcoop.layers import PhysicalGuidedAdaptiveLearnedHypergraph


class PhysCoopLayerTest(unittest.TestCase):
    def test_adaptive_member_selection_and_gradients(self):
        torch.manual_seed(7)
        batch_size, length, num_edges, patch_size, dim = 1, 20, 3, 4, 16
        layer = PhysicalGuidedAdaptiveLearnedHypergraph(
            dim=dim,
            attr_dim=8,
            select_k=8,
            candidate_k=8,
            max_k=6,
            adaptive_members=True,
            min_k=3,
            capacity_init=4,
            use_cooperative_attention=True,
        )
        residue_states = torch.randn(batch_size, length, dim, requires_grad=True)
        ca_coords = torch.randn(batch_size, length, 3) * 8
        residue_mask = torch.ones(batch_size, length, dtype=torch.bool)
        mutation_mask = torch.zeros(batch_size, length, dtype=torch.bool)
        mutation_mask[0, 3] = True
        hyperedge_index = torch.tensor(
            [[[0, 1, 10, 11], [5, 6, 15, 16], [2, 3, 12, 13]]]
        )
        node_mask = torch.ones(batch_size, num_edges, patch_size, dtype=torch.bool)
        edge_mask = torch.ones(batch_size, num_edges, dtype=torch.bool)
        edge_attr = torch.randn(batch_size, num_edges, 8)
        chain_encoding = torch.tensor([[0] * 10 + [1] * 10])

        edge_states, valid_edges, valid_members, _, _ = layer(
            residue_states,
            ca_coords,
            residue_mask,
            mutation_mask,
            hyperedge_index,
            node_mask,
            edge_mask,
            edge_attr,
            chain_encoding=chain_encoding,
        )

        self.assertEqual(edge_states.shape, (batch_size, num_edges, dim))
        self.assertTrue(valid_edges.all())
        self.assertTrue(valid_members.any(dim=-1).all())
        self.assertTrue(torch.all(valid_members.sum(dim=-1) >= 3))
        edge_states.square().mean().backward()
        self.assertIsNotNone(residue_states.grad)
        self.assertGreater(residue_states.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
