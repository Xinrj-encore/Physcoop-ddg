import unittest

import torch

from physcoop.hypergraph import build_contact_delta_sasa_hypergraph


class PhysicalHypergraphTest(unittest.TestCase):
    def test_builds_interface_edges_with_expected_shapes(self):
        length = 12
        positions = torch.zeros(length, 5, 3)
        positions[:6, 1, 0] = torch.linspace(-2.0, 3.0, 6)
        positions[6:, 1, 0] = torch.linspace(-2.0, 3.0, 6)
        positions[6:, 1, 1] = 4.0
        data = {
            "aa": torch.zeros(length, dtype=torch.long),
            "pos_atoms": positions,
            "mask_atoms": torch.ones(length, 5, dtype=torch.bool),
            "chain_nb": torch.tensor([0] * 6 + [1] * 6),
            "mut_flag": torch.tensor([False, False, True, False, False, False] * 2),
            "delta_sasa": torch.linspace(0.0, 30.0, length),
        }

        output = build_contact_delta_sasa_hypergraph(
            data,
            contact_cutoff=8.0,
            patch_radius=10.0,
            max_hyperedges=8,
            max_nodes_per_hyperedge=6,
        )

        self.assertEqual(output["phys_hyperedge_index"].shape, (8, 6))
        self.assertEqual(output["phys_hyperedge_attr"].shape, (8, 8))
        self.assertTrue(output["phys_hyperedge_mask"].any())
        valid_edges = output["phys_hyperedge_mask"]
        valid_nodes = output["phys_hyperedge_node_mask"][valid_edges]
        self.assertTrue(torch.all(valid_nodes.sum(dim=-1) >= 2))


if __name__ == "__main__":
    unittest.main()
