import copy

import torch
from Bio.PDB import Selection
from Bio.PDB.SASA import ShrakeRupley


PHYS_HYPEREDGE_ATTR_DIM = 8


def _residue_key(residue):
    chain_id = residue.get_parent().get_id()
    hetflag, resseq, icode = residue.get_id()
    return chain_id, int(resseq), icode


def _chain_residue_sasa(model, chain_ids):
    """Compute residue SASA for a selected chain set with Bio.PDB."""
    model_copy = copy.deepcopy(model)
    for chain in list(model_copy):
        if chain.get_id() not in chain_ids:
            model_copy.detach_child(chain.get_id())

    sr = ShrakeRupley()
    sr.compute(model_copy, level="R")

    sasa = {}
    for residue in Selection.unfold_entities(model_copy, "R"):
        if hasattr(residue, "sasa"):
            sasa[_residue_key(residue)] = float(residue.sasa)
    return sasa


def compute_delta_sasa(model, seq_map, chain_groups):
    """Return per-residue burial area caused by complex formation.

    delta SASA is approximated as side-alone SASA minus complex SASA for each
    residue in the parsed complex. Values below zero are clipped to zero.
    """
    sr = ShrakeRupley()
    model_complex = copy.deepcopy(model)
    sr.compute(model_complex, level="R")

    complex_sasa = {}
    for residue in Selection.unfold_entities(model_complex, "R"):
        if hasattr(residue, "sasa"):
            complex_sasa[_residue_key(residue)] = float(residue.sasa)

    side_sasa = {}
    for chain_ids in chain_groups:
        if not chain_ids:
            continue
        side_sasa.update(_chain_residue_sasa(model, set(chain_ids)))

    num_residues = max(seq_map.values()) + 1 if seq_map else 0
    delta_sasa = torch.zeros(num_residues, dtype=torch.float32)
    for seq_key, idx in seq_map.items():
        chain_id, resseq = seq_key[:2]
        key = (chain_id, int(resseq), " ")
        if key not in side_sasa and key not in complex_sasa:
            matches = [k for k in complex_sasa if k[0] == chain_id and k[1] == int(resseq)]
            key = matches[0] if matches else key
        buried = side_sasa.get(key, 0.0) - complex_sasa.get(key, 0.0)
        delta_sasa[idx] = max(float(buried), 0.0)
    return delta_sasa


def _get_cb_or_ca_positions(data):
    pos_atoms = data["pos_atoms"]
    mask_atoms = data["mask_atoms"]
    ca = pos_atoms[:, 1]
    if pos_atoms.size(1) < 5:
        return ca
    cb = pos_atoms[:, 4]
    cb_mask = mask_atoms[:, 4].bool().unsqueeze(-1)
    return torch.where(cb_mask, cb, ca)


def _deduplicate_sets(node_sets):
    seen = set()
    unique = []
    for node_set in node_sets:
        key = tuple(sorted(int(i) for i in node_set))
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        unique.append(list(key))
    return unique


def _randomize_hyperedge_sets(hyperedge_sets, num_residues, seed=42):
    """Keep each hyperedge size fixed but replace members with random residues."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    randomized = []
    for node_set in hyperedge_sets:
        count = min(len(node_set), num_residues)
        if count < 2:
            continue
        perm = torch.randperm(num_residues, generator=generator)[:count]
        randomized.append(perm.tolist())
    return randomized


def build_contact_delta_sasa_hypergraph(
    data,
    contact_cutoff=8.0,
    patch_radius=10.0,
    max_hyperedges=96,
    max_nodes_per_hyperedge=24,
    mode="full",
    use_delta_sasa=True,
    use_delta_sasa_attr=True,
    use_mutation_prior=True,
    random_seed=42,
):
    """Build physical interface hyperedges from WT complex geometry.

    Hyperedges are local interface patches seeded by residues with strong
    cross-chain contacts and large burial area. The returned tensors have fixed
    sizes so they can be batched by the existing collate function.
    """
    device = data["aa"].device
    num_residues = data["aa"].size(0)

    edge_index = torch.zeros(
        max_hyperedges, max_nodes_per_hyperedge, dtype=torch.long, device=device
    )
    edge_node_mask = torch.zeros(
        max_hyperedges, max_nodes_per_hyperedge, dtype=torch.bool, device=device
    )
    edge_mask = torch.zeros(max_hyperedges, dtype=torch.bool, device=device)
    edge_attr = torch.zeros(
        max_hyperedges, PHYS_HYPEREDGE_ATTR_DIM, dtype=torch.float32, device=device
    )

    if num_residues < 2:
        return {
            "phys_hyperedge_index": edge_index,
            "phys_hyperedge_node_mask": edge_node_mask,
            "phys_hyperedge_mask": edge_mask,
            "phys_hyperedge_attr": edge_attr,
        }

    coords = _get_cb_or_ca_positions(data)
    chain_nb = data["chain_nb"]
    mut_flag = data.get(
        "mut_flag", torch.zeros(num_residues, dtype=torch.bool, device=device)
    )
    if not use_mutation_prior:
        mut_flag = torch.zeros_like(mut_flag, dtype=torch.bool, device=device)
    delta_sasa = data.get(
        "delta_sasa",
        torch.zeros(num_residues, dtype=torch.float32, device=device),
    )
    delta_sasa = delta_sasa.to(device=device, dtype=torch.float32)
    if not use_delta_sasa:
        delta_sasa = torch.zeros_like(delta_sasa)

    distances = torch.cdist(coords, coords)
    valid_pair = chain_nb.unsqueeze(0) != chain_nb.unsqueeze(1)
    valid_pair.fill_diagonal_(False)
    cross_contact = valid_pair & (distances <= contact_cutoff)

    interface_mask = cross_contact.any(dim=1)
    if mode != "contact_only" and use_delta_sasa:
        interface_mask = interface_mask | (delta_sasa > 1.0)
    seed_indices = torch.nonzero(interface_mask, as_tuple=True)[0]
    if seed_indices.numel() == 0:
        return {
            "phys_hyperedge_index": edge_index,
            "phys_hyperedge_node_mask": edge_node_mask,
            "phys_hyperedge_mask": edge_mask,
            "phys_hyperedge_attr": edge_attr,
        }

    contact_strength = cross_contact.float().sum(dim=1)
    seed_score = contact_strength + mut_flag.float() * 2.0
    if mode != "contact_only" and use_delta_sasa:
        seed_score = seed_score + delta_sasa / 20.0
    seed_indices = seed_indices[torch.argsort(seed_score[seed_indices], descending=True)]

    hyperedge_sets = []
    for seed_idx in seed_indices.tolist():
        local_mask = (distances[seed_idx] <= patch_radius) & interface_mask
        node_indices = torch.nonzero(local_mask, as_tuple=True)[0]
        if node_indices.numel() < 2:
            continue

        chains_in_patch = torch.unique(chain_nb[node_indices])
        if chains_in_patch.numel() < 2:
            continue

        local_score = contact_strength[node_indices]
        if mode != "contact_only" and use_delta_sasa:
            local_score = local_score + delta_sasa[node_indices] / 20.0
        if mut_flag[node_indices].any():
            local_score = local_score + mut_flag[node_indices].float() * 4.0
        keep = min(max_nodes_per_hyperedge, node_indices.numel())
        selected = node_indices[torch.argsort(local_score, descending=True)[:keep]]
        hyperedge_sets.append(selected.tolist())

        if len(hyperedge_sets) >= max_hyperedges * 2:
            break

    hyperedge_sets = _deduplicate_sets(hyperedge_sets)[:max_hyperedges]
    if mode == "random":
        hyperedge_sets = _randomize_hyperedge_sets(
            hyperedge_sets,
            num_residues=num_residues,
            seed=random_seed,
        )

    for edge_id, nodes in enumerate(hyperedge_sets):
        nodes_t = torch.tensor(nodes, dtype=torch.long, device=device)
        count = nodes_t.numel()
        edge_index[edge_id, :count] = nodes_t
        edge_node_mask[edge_id, :count] = True
        edge_mask[edge_id] = True

        sub_dist = distances[nodes_t][:, nodes_t]
        sub_cross = chain_nb[nodes_t].unsqueeze(0) != chain_nb[nodes_t].unsqueeze(1)
        sub_cross = sub_cross & (~torch.eye(count, dtype=torch.bool, device=device))
        cross_dist = sub_dist[sub_cross]
        if cross_dist.numel() == 0:
            mean_contact = torch.tensor(0.0, device=device)
            min_contact = torch.tensor(contact_cutoff, device=device)
            cross_count = torch.tensor(0.0, device=device)
        else:
            within_cutoff = cross_dist <= contact_cutoff
            cross_count = within_cutoff.float().sum() / 2.0
            mean_contact = (
                cross_dist[within_cutoff].mean()
                if within_cutoff.any()
                else cross_dist.mean()
            )
            min_contact = cross_dist.min()

        edge_delta = (
            delta_sasa[nodes_t]
            if use_delta_sasa_attr
            else torch.zeros(count, device=device)
        )
        edge_mut = mut_flag[nodes_t].float()
        edge_attr[edge_id] = torch.stack(
            [
                torch.tensor(float(count) / max_nodes_per_hyperedge, device=device),
                torch.clamp(cross_count / 32.0, max=1.0),
                torch.clamp(1.0 - mean_contact / contact_cutoff, min=0.0, max=1.0),
                torch.clamp(1.0 - min_contact / contact_cutoff, min=0.0, max=1.0),
                torch.clamp(edge_delta.mean() / 50.0, max=1.0),
                torch.clamp(edge_delta.max() / 100.0, max=1.0),
                edge_mut.max(),
                edge_mut.mean(),
            ]
        )

    return {
        "phys_hyperedge_index": edge_index,
        "phys_hyperedge_node_mask": edge_node_mask,
        "phys_hyperedge_mask": edge_mask,
        "phys_hyperedge_attr": edge_attr,
    }
