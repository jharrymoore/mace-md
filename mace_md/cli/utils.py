from enum import Enum


class Solvents(Enum):
    TIP3P = "tip3p"
    TIP4P = "tip4p"
    OCTANOL = "octanol"


class RunType(Enum):
    MD = "md"
    REPEX = "repex"
