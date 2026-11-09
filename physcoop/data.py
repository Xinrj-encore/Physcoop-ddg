import copy
import functools
import math
import os
import pickle

import torch
from Bio.PDB.PDBParser import PDBParser
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from . import _bio_compat  # noqa: F401
from common_utils.protein.parsers import parse_biopython_structure
from skempi import SkempiDataset

from .hypergraph import build_contact_delta_sasa_hypergraph, compute_delta_sasa


HYPERGRAPH_KEYS = {
    "phys_hyperedge_index",
    "phys_hyperedge_node_mask",
    "phys_hyperedge_mask",
    "phys_hyperedge_attr",
}


class PhysCoopSkempiDataset(SkempiDataset):
    def __init__(
        self,
        csv_path,
        pdb_dir,
        cache_dir,
        num_cvfolds=3,
        cvfold_index=0,
        split="train",
        reset=False,
        contact_cutoff=8.0,
        patch_radius=10.0,
        max_hyperedges=96,
        max_nodes_per_hyperedge=24,
        hypergraph_mode="full",
        use_delta_sasa=True,
        use_delta_sasa_attr=True,
        use_mutation_prior=True,
    ):
        self.contact_cutoff = float(contact_cutoff)
        self.patch_radius = float(patch_radius)
        self.max_hyperedges = int(max_hyperedges)
        self.max_nodes_per_hyperedge = int(max_nodes_per_hyperedge)
        self.hypergraph_mode = str(hypergraph_mode)
        self.use_delta_sasa = bool(use_delta_sasa)
        self.use_delta_sasa_attr = bool(use_delta_sasa_attr)
        self.use_mutation_prior = bool(use_mutation_prior)
        super().__init__(
            csv_path=csv_path,
            pdb_dir=pdb_dir,
            cache_dir=cache_dir,
            num_cvfolds=num_cvfolds,
            cvfold_index=cvfold_index,
            split=split,
            reset=reset,
        )
        self._ensure_delta_sasa()

    def _ensure_delta_sasa(self):
        has_delta_sasa = all(
            isinstance(value, tuple) and value and "delta_sasa" in value[0]
            for value in self.structures.values()
        )
        if has_delta_sasa:
            return

        parser = PDBParser(QUIET=True)
        structures = {}
        complexes = sorted({entry["complex"] for entry in self.entries_full})
        for complex_id in tqdm(complexes, desc="Delta SASA"):
            pdb_id = complex_id.split("_")[0]
            antibody_chains = list(complex_id.split("_")[-1])
            antigen_chains = list(complex_id.split("_")[-2])
            pdb_path = os.path.join(self.pdb_dir, f"{pdb_id.upper()}.pdb")
            model = parser.get_structure(None, pdb_path)[0]
            data, seq_map = parse_biopython_structure(
                model,
                antibody_chain_id=antibody_chains,
                antigen_chain_id=antigen_chains,
            )
            data["delta_sasa"] = compute_delta_sasa(
                model,
                seq_map,
                chain_groups=[antibody_chains, antigen_chains],
            )
            structures[pdb_id] = (data, seq_map)

        self.structures = structures
        with open(self.structures_cache, "wb") as handle:
            pickle.dump(structures, handle)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        entry = self.entries[index]
        data.update(
            build_contact_delta_sasa_hypergraph(
                data,
                contact_cutoff=self.contact_cutoff,
                patch_radius=self.patch_radius,
                max_hyperedges=self.max_hyperedges,
                max_nodes_per_hyperedge=self.max_nodes_per_hyperedge,
                mode=self.hypergraph_mode,
                use_delta_sasa=self.use_delta_sasa,
                use_delta_sasa_attr=self.use_delta_sasa_attr,
                use_mutation_prior=self.use_mutation_prior,
                random_seed=int(entry["id"]) + 42,
            )
        )
        return data


class PhysCoopCollate:
    pad_values = {
        "aa": 21,
        "aa_mut": 21,
        "chain_encoding_all": 0,
        "residue_idx": -100,
        "mask": 0,
        "chain_M": 0,
        "delta_sasa": 0.0,
    }

    @staticmethod
    def _pad_last(value, target_length, fill_value=0):
        if not isinstance(value, torch.Tensor) or value.dim() == 0:
            return value
        if value.size(0) >= target_length:
            return value
        padding = torch.full(
            (target_length - value.size(0), *value.shape[1:]),
            fill_value=fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat([value, padding], dim=0)

    @staticmethod
    def _reset_residue_idx(residue_numbers):
        reset_points = (residue_numbers == 1).nonzero(as_tuple=True)[0][1:]
        offsets = torch.zeros_like(residue_numbers)
        offsets[reset_points] = 100 + residue_numbers[reset_points - 1]
        return residue_numbers + torch.cumsum(offsets, dim=0)

    def __call__(self, samples):
        max_length = max(sample["aa"].size(0) for sample in samples)
        max_length = math.ceil(max_length / 8) * 8
        batch_samples = []
        num_mutation_chains = []

        for sample in samples:
            model_input = {
                "X": sample["pos_atoms"][..., :4, :],
                "aa": sample["aa"],
                "aa_mut": sample["aa_mut"],
                "mask": torch.ones_like(sample["aa"]),
                "chain_M": sample["mut_flag"],
                "chain_encoding_all": sample["chain_nb"] + 1,
                "residue_idx": self._reset_residue_idx(sample["res_nb"]),
                "complex": sample["complex"],
                "ddG": sample["ddG"],
                "id": sample["id"],
                "num_muts": sample["num_muts"],
                "mutstr": sample["mutstr"],
                "delta_sasa": sample.get(
                    "delta_sasa", torch.zeros_like(sample["aa"], dtype=torch.float32)
                ),
            }
            for key in HYPERGRAPH_KEYS:
                model_input[key] = sample[key]

            padded = {}
            for key, value in model_input.items():
                if key in HYPERGRAPH_KEYS:
                    padded[key] = value
                else:
                    padded[key] = self._pad_last(
                        value, max_length, self.pad_values.get(key, 0)
                    )
            batch_samples.append(padded)

            mutation_indices = torch.nonzero(sample["mut_flag"], as_tuple=True)[0]
            mutation_chains = torch.unique(
                padded["chain_encoding_all"][mutation_indices]
            )
            num_mutation_chains.append(mutation_chains.numel())

            for chain_id in mutation_chains.tolist():
                chain_mask = padded["chain_encoding_all"] == chain_id
                single_chain = copy.deepcopy(padded)
                single_chain["mask"] = chain_mask
                single_chain["chain_M"] = chain_mask * padded["chain_M"]
                single_chain["X"] = single_chain["X"] * chain_mask[:, None, None]
                single_chain["aa"] = single_chain["aa"] * chain_mask
                single_chain["aa_mut"] = single_chain["aa_mut"] * chain_mask
                single_chain["delta_sasa"] = single_chain["delta_sasa"] * chain_mask.to(
                    single_chain["delta_sasa"].dtype
                )
                batch_samples.append(single_chain)

        batch = default_collate(batch_samples)
        batch["num_mut_chains"] = num_mutation_chains
        return batch


def _infinite_iterator(iterable):
    while True:
        yield from iterable


class PhysCoopDatasetManager:
    def __init__(self, config, num_cvfolds, num_workers=4):
        self.config = config
        self.num_cvfolds = num_cvfolds
        self.num_workers = num_workers
        self.train_loaders = []
        self.val_loaders = []
        for fold in range(num_cvfolds):
            train_loader, val_loader, all_loader = self._build_loaders(fold)
            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)
        self.all_loader = all_loader

    def _build_loaders(self, fold):
        dataset_factory = functools.partial(
            PhysCoopSkempiDataset,
            csv_path=getattr(
                self.config, "skempi_csv_path", "./SKEMPI_v2/skempi_v2.csv"
            ),
            pdb_dir=getattr(self.config, "skempi_pdb_dir", "./SKEMPI_v2/PDBs"),
            cache_dir=getattr(
                self.config, "skempi_cache_dir", "./SKEMPI_v2_cache_phys"
            ),
            num_cvfolds=self.num_cvfolds,
            cvfold_index=fold,
            contact_cutoff=getattr(self.config, "phys_contact_cutoff", 8.0),
            patch_radius=getattr(self.config, "phys_patch_radius", 10.0),
            max_hyperedges=getattr(self.config, "phys_max_hyperedges", 96),
            max_nodes_per_hyperedge=getattr(
                self.config, "phys_max_nodes_per_hyperedge", 24
            ),
            hypergraph_mode=getattr(self.config, "phys_hypergraph_mode", "full"),
            use_delta_sasa=getattr(self.config, "phys_use_delta_sasa", True),
            use_delta_sasa_attr=getattr(
                self.config, "phys_use_delta_sasa_attr", True
            ),
            use_mutation_prior=getattr(
                self.config, "phys_use_mutation_prior", True
            ),
        )
        train_dataset = dataset_factory(split="train")
        val_dataset = dataset_factory(split="val")
        all_dataset = dataset_factory(split="all")

        train_complexes = {entry["complex"] for entry in train_dataset.entries}
        val_complexes = {entry["complex"] for entry in val_dataset.entries}
        leakage = train_complexes & val_complexes
        if leakage:
            raise RuntimeError(f"Complex-level leakage detected: {sorted(leakage)}")

        collate = PhysCoopCollate()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate,
            num_workers=self.num_workers,
        )
        all_loader = DataLoader(
            all_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.num_workers,
        )
        print(
            f"Fold {fold}: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, all={len(all_dataset)}"
        )
        return _infinite_iterator(train_loader), val_loader, all_loader

    def get_train_loader(self, fold):
        return self.train_loaders[fold]

    def get_val_loader(self, fold):
        return self.val_loaders[fold]
