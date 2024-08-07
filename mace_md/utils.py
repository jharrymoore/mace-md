from typing import Optional, Union
import logging
import sys
import os
from argparse import ArgumentParser

from mace_md.hybrid_md import ReplicaMixingScheme


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + ".log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)


def parse_arguments():

    parser = ArgumentParser()

    parser.add_argument("--file", "-f", type=str)
    parser.add_argument(
        "--ml_mol",
        type=str,
        help="either smiles string or file path for the \
            small molecule to be described by MACE",
        default=None,
    )
    parser.add_argument(
        "--run_type", choices=["md", "repex", "neq", "atm"], type=str, default="md"
    )
    parser.add_argument("--steps", "-s", type=int, default=10000)
    parser.add_argument("--steps_per_iter", "-spi", type=int, default=1000)
    parser.add_argument("--padding", "-p", default=1.2, type=float)
    parser.add_argument(
        "--box_shape",
        type=str,
        default="cube",
        choices=["cube", "dodecahedron", "octahedron"],
    )
    parser.add_argument("--constrain_res", type=str, default=None)
    parser.add_argument("--nonbondedCutoff", "-c", default=1.0, type=float)
    parser.add_argument("--ionic_strength", "-i", default=0.15, type=float)
    parser.add_argument("--potential", default="mace", type=str)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument(
        "--minimiser", type=str, choices=["openmm", "ase"], default=None
    )
    parser.add_argument("--pressure", type=float, default=None)
    parser.add_argument("--remove_cmm", action="store_true")
    parser.add_argument("--set_temperature", action="store_true")
    parser.add_argument(
        "--unwrap",
        action="store_true",
        help="Control whether the reporters write unwrapped coordinates (useful for materials systems with no molecules)",
    )
    parser.add_argument(
        "--integrator",
        type=str,
        default="langevin",
        choices=["langevin", "nose-hoover", "rpmd", "verlet"],
    )
    parser.add_argument(
        "--timestep",
        default=1.0,
        help="integration timestep in femtoseconds",
        type=float,
    )
    parser.add_argument(
        "--extract_nb",
        action="store_true",
        help="If true, extracts non-bonded components of the SM forcefield, adds them \
            to a separate array on the atoms object, writes back out",
    )
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument(
        "--replica_mixing_scheme",
        type=ReplicaMixingScheme,
        default=ReplicaMixingScheme.SWAP_ALL,
    )
    parser.add_argument("--lambda_schedule", type=str, default=None)
    parser.add_argument("--optimized_model", action="store_true")
    parser.add_argument(
        "--direction", type=str, choices=["forward", "reverse"], default="forward"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="output.pdb",
        help="output file for the pdb reporter",
    )
    parser.add_argument("--log_level", default="INFO", type=str)
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--output_dir",
        help="directory where all output will be written",
        default="./junk",
    )

    # optionally specify box vectors for periodic systems
    parser.add_argument("--box", type=float)

    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--decouple",
        help="tell the repex constructor to deal with decoupling sterics + \
            electrostatics, instead of lambda_interpolate",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--equil", type=str, choices=["minimise", "gentle"], default="minimise"
    )
    parser.add_argument(
        "--forcefields",
        type=list,
        default=[
            "amber/protein.ff14SB.xml",
            "amber14/DNA.OL15.xml",
        ],
    )
    parser.add_argument("--water_model", type=str, default="tip3p")
    parser.add_argument(
        "--smff",
        help="which version of the openff small molecule forcefield to use",
        default="1.0",
        type=str,
        choices=["1.0", "2.0", "2.0-constrained"],
    )
    parser.add_argument(
        "--interval", help="steps between saved frames", type=int, default=100
    )
    parser.add_argument(
        "--resname",
        "-r",
        help="name of the ligand residue in pdb file",
        default="UNK",
        type=str,
    )
    parser.add_argument(
        "--nl",
        help="which neighbour list to use",
        choices=["nnpops", "torch"],
        default="nnpops",
    )
    parser.add_argument(
        "--max_n_pairs",
        help="maximum number of pairs to return for the neighbour list",
        type=int,
        default=-1,
    )
    parser.add_argument("--meta", help="Switch on metadynamics", action="store_true")
    parser.add_argument(
        "--model_path",
        "-m",
        help="path to the mace model",
        default="tests/test_openmm/MACE_SPICE_larger.model",
    )
    parser.add_argument(
        "--system_type",
        type=str,
        choices=["pure", "hybrid", "decoupled"],
        default="pure",
    )
    parser.add_argument("--mm_only", action="store_true", default=False)
    parser.add_argument("--write_gmx", action="store_true", default=False)
    parser.add_argument(
        "--ml_selection",
        help="specify how the ML subset should be interpreted, \
            either as a resname or a chain ",
        choices=["resname", "chain"],
        default="resname",
    )
    return parser
