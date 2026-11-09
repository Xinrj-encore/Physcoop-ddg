"""BindingGYM inference adapter.

BindingGYM stores mutations in a per-chain dictionary and may use insertion
codes, for example ``P52AL``.  This module resolves those mutations against
the parsed PDB residue identity instead of relying on the SKEMPI two-field
``(chain, resseq)`` lookup.
"""

from __future__ import annotations

import ast
import copy
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from torch.utils.data import Dataset

from . import _bio_compat  # noqa: F401
from common_utils.protein.constants import resindex_to_ressymb, ressymb_to_resindex
from common_utils.protein.parsers import parse_biopython_structure
from common_utils.transforms import get_transform
from dataset import MPNNPaddingCollate
from .hypergraph import (
    build_contact_delta_sasa_hypergraph,
    compute_delta_sasa,
)


MUTATION_RE = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)([A-Za-z])$")


def _parse_mutation_dict(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return {}
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        raise ValueError("BindingGYM mutation field is not a dictionary.")
    return parsed


def parse_bindinggym_mutations(value):
    """Parse BindingGYM ``mutant_pdb`` into chain-aware mutation records."""
    mutations = []
    mutation_dict = _parse_mutation_dict(value)
    for chain, chain_value in mutation_dict.items():
        chain = str(chain).strip()
        if not chain:
            continue
        chain_value = "" if chain_value is None else str(chain_value).strip()
        if not chain_value:
            continue
        for token in chain_value.split(":"):
            token = token.strip()
            if not token:
                continue
            match = MUTATION_RE.fullmatch(token)
            if match is None:
                raise ValueError(f"Unsupported BindingGYM mutation token: {token!r}")
            wt, resseq, icode, mt = match.groups()
            mutations.append(
                {
                    "chain": chain,
                    "wt": wt.upper(),
                    "resseq": int(resseq),
                    "icode": icode.upper() if icode else " ",
                    "mt": mt.upper(),
                    "token": token,
                }
            )
    return mutations


def _exact_residue_map(data):
    return {
        (str(chain), int(resseq), str(icode) or " "): index
        for index, (chain, resseq, icode) in enumerate(
            zip(data["chain_id"], data["resseq"], data["icode"])
        )
    }


def _resolve_residue_index(seq_map, mutation):
    key = (mutation["chain"], mutation["resseq"], mutation["icode"])
    if key in seq_map:
        return seq_map[key]

    if mutation["icode"] == " ":
        candidates = [
            index
            for (chain, resseq, _icode), index in seq_map.items()
            if chain == mutation["chain"] and resseq == mutation["resseq"]
        ]
        if len(candidates) == 1:
            return candidates[0]
    return None


def _format_mutation(mutation):
    insertion_code = mutation["icode"].strip()
    return (
        f"{mutation['wt']}{mutation['chain']}{mutation['resseq']}"
        f"{insertion_code}{mutation['mt']}"
    )


class BindingGymDataset(Dataset):
    """Structure-resolved BindingGYM samples for zero-shot inference."""

    def __init__(
        self,
        manifest_path,
        input_dir,
        use_physical_hypergraph=True,
        phys_contact_cutoff=8.0,
        phys_patch_radius=10.0,
        phys_max_hyperedges=96,
        phys_max_nodes_per_hyperedge=24,
        phys_hypergraph_mode="full",
        phys_use_delta_sasa=True,
        phys_use_delta_sasa_attr=True,
        assay_ids=None,
        max_samples=None,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.input_dir = Path(input_dir)
        self.use_physical_hypergraph = bool(use_physical_hypergraph)
        self.phys_contact_cutoff = float(phys_contact_cutoff)
        self.phys_patch_radius = float(phys_patch_radius)
        self.phys_max_hyperedges = int(phys_max_hyperedges)
        self.phys_max_nodes_per_hyperedge = int(phys_max_nodes_per_hyperedge)
        self.phys_hypergraph_mode = str(phys_hypergraph_mode)
        self.phys_use_delta_sasa = bool(phys_use_delta_sasa)
        self.phys_use_delta_sasa_attr = bool(phys_use_delta_sasa_attr)
        self.transform = get_transform(
            [{"type": "select_atom", "resolution": "backbone+CB"}]
        )
        self._structure_cache = {}

        frame = pd.read_csv(self.manifest_path, low_memory=False)
        required = {
            "sample_id",
            "assay_id",
            "dms_filename",
            "row_index",
            "poi",
            "complex_id",
            "pdb_file",
            "chain_id",
            "dms_score",
            "mutation_pdb",
            "num_muts",
            "status",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"BindingGYM manifest is missing columns: {missing}")

        frame = frame[frame["status"].astype(str).str.lower().eq("ok")].copy()
        if assay_ids:
            assay_ids = {str(assay_id) for assay_id in assay_ids}
            frame = frame[frame["assay_id"].astype(str).isin(assay_ids)].copy()
        if max_samples is not None:
            frame = frame.head(int(max_samples)).copy()
        if frame.empty:
            raise ValueError("No structure-resolved BindingGYM rows remain after filtering.")

        frame["_dataset_index"] = np.arange(len(frame), dtype=np.int64)
        self.frame = frame.reset_index(drop=True)
        self.assay_mutated_chains = self._build_assay_mutated_chains(self.frame)

    def __len__(self):
        return len(self.frame)

    def prepare_structure_metadata(self):
        """Preload structures and annotate row-level resource metadata."""
        metadata_by_key = {}
        group_columns = ["pdb_file", "chain_id", "assay_id"]
        for key, group in self.frame.groupby(group_columns, sort=False):
            row = group.iloc[0]
            data, _exact_seq_map = self._load_structure(row)
            groups = data.get("bindinggym_chain_groups")
            if groups is None:
                actual_chains = sorted({str(chain) for chain in data["chain_id"]})
                groups = [[chain] for chain in actual_chains]
            metadata_by_key[key] = (
                int(data["aa"].size(0)),
                "|".join("".join(sorted(group)) for group in groups),
            )

        row_keys = list(
            zip(
                self.frame["pdb_file"],
                self.frame["chain_id"],
                self.frame["assay_id"],
            )
        )
        self.frame["num_residues"] = [metadata_by_key[key][0] for key in row_keys]
        self.frame["bindinggym_chain_groups"] = [
            metadata_by_key[key][1] for key in row_keys
        ]

    def _build_assay_mutated_chains(self, frame):
        assay_mutated_chains = {}
        for assay_id, assay in frame.groupby("assay_id", sort=False):
            mutated_chains = set()
            for value in assay["mutation_pdb"]:
                for mutation in parse_bindinggym_mutations(value):
                    mutated_chains.add(mutation["chain"])
            assay_mutated_chains[str(assay_id)] = mutated_chains
        return assay_mutated_chains

    def _chain_groups_for_row(self, row, actual_chains):
        mutated_chains = set(self.assay_mutated_chains.get(str(row["assay_id"]), set()))
        actual_chain_set = set(actual_chains)
        mutated_chains = sorted(mutated_chains.intersection(actual_chain_set))
        partner_chains = sorted(actual_chain_set.difference(mutated_chains))
        if mutated_chains and partner_chains:
            return [mutated_chains, partner_chains]
        return [[chain] for chain in actual_chains]

    def _structure_key(self, row):
        assay_chains = ",".join(
            sorted(self.assay_mutated_chains.get(str(row["assay_id"]), set()))
        )
        return (str(row["pdb_file"]), str(row["chain_id"]), assay_chains)

    def _load_structure(self, row):
        cache_key = self._structure_key(row)
        if cache_key in self._structure_cache:
            return self._structure_cache[cache_key]

        pdb_file = str(row["pdb_file"])
        pdb_path = self.input_dir / "structures" / pdb_file
        if not pdb_path.exists():
            raise FileNotFoundError(f"BindingGYM structure not found: {pdb_path}")

        requested_chains = set(str(row["chain_id"]).strip())
        parser = PDBParser(QUIET=True)
        model = parser.get_structure(pdb_file, str(pdb_path))[0]
        model = copy.deepcopy(model)
        for chain in list(model):
            if chain.get_id() not in requested_chains:
                model.detach_child(chain.get_id())

        data, _unused_seq_map = parse_biopython_structure(model)
        if data is None:
            raise ValueError(f"No parseable protein residues in {pdb_path}")

        actual_chains = sorted({str(chain) for chain in data["chain_id"]})
        missing_chains = sorted(requested_chains.difference(actual_chains))
        if missing_chains:
            raise ValueError(
                f"Chains {missing_chains} from manifest are absent after parsing {pdb_path}; "
                f"parsed chains={actual_chains}"
            )

        exact_seq_map = _exact_residue_map(data)
        if self.use_physical_hypergraph:
            chain_groups = self._chain_groups_for_row(row, actual_chains)
            data["delta_sasa"] = compute_delta_sasa(
                model,
                exact_seq_map,
                chain_groups=chain_groups,
            )
            data["bindinggym_chain_groups"] = chain_groups

        cached = (data, exact_seq_map)
        self._structure_cache[cache_key] = cached
        return cached

    def __getitem__(self, index):
        row = self.frame.iloc[int(index)]
        base_data, exact_seq_map = self._load_structure(row)
        data = copy.deepcopy(base_data)

        mutations = parse_bindinggym_mutations(row["mutation_pdb"])
        if not mutations:
            raise ValueError(f"BindingGYM row has no mutations: {row['sample_id']}")

        aa_mut = data["aa"].clone()
        resolved_indices = []
        for mutation in mutations:
            residue_index = _resolve_residue_index(exact_seq_map, mutation)
            if residue_index is None:
                raise ValueError(
                    f"Cannot resolve {_format_mutation(mutation)} in "
                    f"{row['pdb_file']} for sample {row['sample_id']}"
                )

            observed = resindex_to_ressymb[int(data["aa"][residue_index])]
            if mutation["wt"] != "X" and observed != mutation["wt"]:
                raise ValueError(
                    f"WT mismatch for {_format_mutation(mutation)} in "
                    f"{row['pdb_file']}: structure has {observed}"
                )
            if mutation["mt"] not in ressymb_to_resindex:
                raise ValueError(
                    f"Unsupported mutant residue {mutation['mt']!r} "
                    f"in sample {row['sample_id']}"
                )
            aa_mut[residue_index] = ressymb_to_resindex[mutation["mt"]]
            resolved_indices.append(residue_index)

        if len(set(resolved_indices)) != len(resolved_indices):
            raise ValueError(f"Duplicate mutation position in sample {row['sample_id']}")

        data["aa_mut"] = aa_mut
        data["mut_flag"] = data["aa"] != data["aa_mut"]
        data["ddG"] = np.float32(row["dms_score"])
        data["dG_wt"] = np.float32(0.0)
        data["dG_mut"] = np.float32(0.0)
        data["temperature"] = np.float32(298.15)
        data["id"] = int(row["_dataset_index"])
        data["complex"] = str(row["complex_id"])
        data["num_muts"] = int(len(mutations))
        data["mutstr"] = ",".join(_format_mutation(mutation) for mutation in mutations)

        if self.transform is not None:
            data = self.transform(data)

        if self.use_physical_hypergraph:
            data.update(
                build_contact_delta_sasa_hypergraph(
                    data,
                    contact_cutoff=self.phys_contact_cutoff,
                    patch_radius=self.phys_patch_radius,
                    max_hyperedges=self.phys_max_hyperedges,
                    max_nodes_per_hyperedge=self.phys_max_nodes_per_hyperedge,
                    mode=self.phys_hypergraph_mode,
                    use_delta_sasa=self.phys_use_delta_sasa,
                    use_delta_sasa_attr=self.phys_use_delta_sasa_attr,
                    random_seed=int(row["_dataset_index"]) + 42,
                )
            )

        data["sample_id"] = str(row["sample_id"])
        data["assay_id"] = str(row["assay_id"])
        return data


class BindingGymCollate(MPNNPaddingCollate):
    """Preserve row metadata while reusing the model's thermodynamic collate."""

    def __call__(self, data_list):
        batch = super().__call__(data_list)
        batch["sample_id"] = [str(data["sample_id"]) for data in data_list]
        batch["assay_id"] = [str(data["assay_id"]) for data in data_list]
        return batch
