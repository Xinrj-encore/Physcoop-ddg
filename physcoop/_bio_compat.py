from Bio.PDB import Polypeptide


if not hasattr(Polypeptide, "three_to_one"):
    mapping = getattr(Polypeptide, "protein_letters_3to1", {})

    def three_to_one(resname):
        try:
            return mapping[resname]
        except KeyError as exc:
            raise KeyError(f"Unknown three-letter residue code: {resname}") from exc

    Polypeptide.three_to_one = three_to_one


if not hasattr(Polypeptide, "one_to_three"):
    inverse_mapping = {
        one: three
        for three, one in getattr(Polypeptide, "protein_letters_3to1", {}).items()
    }

    def one_to_three(resname):
        try:
            return inverse_mapping[resname]
        except KeyError as exc:
            raise KeyError(f"Unknown one-letter residue code: {resname}") from exc

    Polypeptide.one_to_three = one_to_three
