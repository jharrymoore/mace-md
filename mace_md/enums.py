from dataclasses import dataclass


class ReplicaMixingScheme:
    SWAP_ALL = "swap-all"
    SWAP_NONE = None
    SWAP_NEIGHBORS = "swap-neighbors"


SOLVENT_SMILES = {
    "octanol": "CCCCCCCCO",
}
