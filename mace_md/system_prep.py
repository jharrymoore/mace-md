from typing import Literal, Optional
import errno

from random import randint
import subprocess
import os
import shutil
import functools

from openff.toolkit.topology import Molecule
from openmm.app import Element, Modeller, PDBFile, Topology
from openmmtools.states import reduced_potential_at_states
from openmm import unit
from ase.io import read
import logging
import openmm
import numpy as np
import openff

from rdkit import Chem

from mace_md.nnp_repex.utils import initialize_mm_forcefield

_G_PER_ML = openmm.unit.grams / openmm.unit.milliliters
_G_PER_MOLE = openmm.unit.grams / openmm.unit.mole


def get_xyz_from_mol(mol: Molecule) -> np.ndarray:
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return xyz


def modeller_from_xyz(
    file: str, padding: float, box_shape: Literal["cube", "dodecahedron"]
) -> Modeller:

    atoms = read(file)
    pos = atoms.get_positions() / 10
    box_vectors = atoms.get_cell() / 10
    # canonicalise
    if max(atoms.get_cell().cellpar()[:3]) > 0:
        box_vectors = reduced_potential_at_states(box_vectors)
    logging.info(f"Using reduced periodic box vectors {box_vectors}")
    elements = atoms.get_chemical_symbols()
    molecule = Molecule

    # Create a topology object
    topology = Topology()

    # Add atoms to the topology
    chain = topology.addChain()
    res = topology.addResidue("mace_system", chain)
    for i, (element, position) in enumerate(zip(elements, pos)):
        e = Element.getBySymbol(element)
        topology.addAtom(str(i), e, res)
    # if there is a periodic box specified add it to the Topology
    if max(atoms.get_cell().cellpar()[:3]) > 0:
        topology.setPeriodicBoxVectors(vectors=box_vectors)
    modeller = Modeller(topology, pos)

    if padding > 0:
        # this parsing is prone to fail, only attempt if solvation is requested
        rdmol = Chem.MolFromXYZFile(file)
        molecule = Molecule.from_rdkit(rdmol)
        modeller = solvate_system(molecule, modeller, padding, box_shape)
    return modeller


def modeller_from_sdf(
    file: str, padding: float, box_shape: Literal["cube", "dodecahedron"]
) -> Modeller:

    molecule = Molecule.from_file(file)

    # input_file = molecule
    topology = molecule.to_topology().to_openmm()
    # Hold positions in nanometers
    positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

    logging.info(f"Initialized topology with {positions.shape} positions")

    modeller = Modeller(topology, positions)
    if padding > 0:
        modeller = solvate_system(molecule, modeller, padding, box_shape)
    return modeller


def modeller_from_pdb(file: str, padding: float, box_shape) -> Modeller:

    pdb = PDBFile(file)
    topology = pdb.getTopology()
    positions = pdb.getPositions()
    modeller = Modeller(topology, positions)
    if padding > 0:
        modeller = solvate_system(None, modeller, padding, box_shape)

    return modeller


def modeller_from_smiles(
    file: str, padding: float, box_shape: Literal["cube", "dodecahedron"]
) -> Modeller:

    molecule = Molecule.from_smiles(file)
    molecule.generate_conformers()
    topology = molecule.to_topology().to_openmm()
    positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

    modeller = Modeller(molecule.to_topology().to_openmm(), positions)
    if padding > 0:
        modeller = solvate_system(molecule, modeller, padding, box_shape)
    return modeller


def solvate_system(
    molecule: Optional[Molecule],
    modeller: Modeller,
    padding: float,
    box_shape: Literal["cube", "dodecahedron"],
) -> Modeller:

    logging.info("Solvating system...")
    # require the forcefield for the modeller only
    forcefield = initialize_mm_forcefield(molecule)
    modeller.addSolvent(
        forcefield=forcefield,
        model="tip3p",
        padding=padding * openmm.unit.nanometers,
        boxShape=box_shape,
        ionicStrength=0 * openmm.unit.molar,
        neutralize=False,
    )
    return modeller


def modeller_from_packmol(
    components: list[tuple[str, int]],
    box_target_density: openmm.unit.Quantity = 0.95 * _G_PER_ML,
    box_scale_factor: float = 1.0,
    box_padding: openmm.unit.Quantity = 0.5 * openmm.unit.angstrom,
    tolerance: openmm.unit.Quantity = 1.5 * openmm.unit.angstrom,
) -> Modeller:
    """Generate a set of molecule coordinate by using the PACKMOL package.

    Args:
        components: A list of the form ``components[i] = (smiles_i, count_i)`` where
            ``smiles_i`` is the SMILES representation of component `i` and
            ``count_i`` is the number of corresponding instances of that component
            to create.
        box_target_density: Target mass density when approximating the box size for the
            final system with units compatible with g / mL.
        box_scale_factor: The amount to scale the approximate box size by.
        box_padding: The amount of extra padding to add to the box size to avoid PBC
            issues in units compatible with angstroms.
        tolerance: The minimum spacing between molecules during packing in units
             compatible with angstroms.

    Returns:
        A topology containing the molecules the coordinates were generated for and
        a unit [A] wrapped numpy array of coordinates with shape=(n_atoms, 3).
    """

    packmol_path = shutil.which("packmol")

    if packmol_path is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "packmol")

    box_size = (
        _approximate_box_size_by_density(components, box_target_density)
        * box_scale_factor
    ) * openmm.unit.angstrom
    logging.info(f"Approximated box size: {box_size} for density {box_target_density}")
    molecules = {}

    for (smiles, _) in components:
        if smiles in molecules:
            continue

        molecule = openff.toolkit.Molecule.from_smiles(smiles)
        molecule.generate_conformers(n_conformers=1)
        molecule.name = f"component-{len(molecules)}.xyz"
        molecules[smiles] = molecule

    with openff.utilities.temporary_cd():
        for molecule in molecules.values():
            molecule.to_file(molecule.name, "xyz")

        input_file_contents = _generate_input_file(
            [(molecules[smiles].name, count) for smiles, count in components],
            box_size,
            tolerance,
        )
        print(input_file_contents)
        with open("input.txt", "w") as file:
            file.write(input_file_contents)

        with open("input.txt") as file:
            subprocess.run(packmol_path, stdin=file, check=True, capture_output=True)

        with open("output.xyz") as file:
            output_lines = file.read().splitlines(False)

    coordinates = (
        np.array(
            [
                [float(coordinate) for coordinate in coordinate_line.split()[1:]]
                for coordinate_line in output_lines[2:]
                if len(coordinate_line) > 0
            ]
        )
        * openmm.unit.angstrom
    )

    topology = openff.toolkit.Topology.from_molecules(
        [molecules[smiles] for smiles, count in components for _ in range(count)]
    )
    # change the resname for the first residue to LIG
    topology.box_vectors = np.eye(3) * (box_size + box_padding)
    topology = topology.to_openmm()
    for res in topology.residues():
        print(res.name)
        res.name = "LIG"
        break

    modeller = Modeller(topology, coordinates)

    return modeller


def compute_num_molecules(w_A, MW_A, w_B, MW_B, density, box_volume):
    """
    Compute molecule numbers for a given box size and density
    """
    N_Avogadro = 6.02214076e23  # molecules/mol

    M_total = density * box_volume  # grams

    M_A = w_A * M_total  # grams
    M_B = w_B * M_total  # grams

    n_A = M_A / MW_A  # moles
    n_B = M_B / MW_B  # moles

    N_A = n_A * N_Avogadro
    N_B = n_B * N_Avogadro

    return N_A, N_B


def _approximate_num_molecules_by_density(
    components: list[str],
    padding: openmm.unit.Quantity,
    target_density: openmm.unit.Quantity,
) -> openmm.unit.Quantity:
    """Generate an approximate box size based on the number and molecular weight of
    the molecules present, and a target density for the final system.

    Args:
        components: The list of components.
        target_density: Target mass density for final system with units compatible
            with g / mL.

    Returns:
        The box size.
    """
    target_volume = (2.0 * padding) ** 3 * unit.nanometers**3

    molecules = {
        smiles: openff.toolkit.Molecule.from_smiles(smiles) for smiles in components
    }

    # TODO - this will only work for single components at the moment
    for smiles in components:
        molecule_mass = functools.reduce(
            (lambda x, y: x + y),
            [
                atom.mass.to_openmm().value_in_unit(_G_PER_MOLE)
                for atom in molecules[smiles].atoms
            ],
        )
        molecule_mass /= 6.02214076e23
        molecule_volume = molecule_mass / target_density * unit.centimeter**3

    return int(target_volume / molecule_volume)


def _approximate_box_size_by_density(
    components: list[tuple[str, int]],
    target_density: openmm.unit.Quantity,
) -> openmm.unit.Quantity:
    """Generate an approximate box size based on the number and molecular weight of
    the molecules present, and a target density for the final system.

    Args:
        components: The list of components.
        target_density: Target mass density for final system with units compatible
            with g / mL.

    Returns:
        The box size.
    """

    molecules = {
        smiles: openff.toolkit.Molecule.from_smiles(smiles)
        for smiles in {smiles for smiles, _ in components}
    }

    volume = 0.0

    for smiles, count in components:
        molecule_mass = functools.reduce(
            (lambda x, y: x + y),
            [
                atom.mass.to_openmm().value_in_unit(_G_PER_MOLE)
                for atom in molecules[smiles].atoms
            ],
        )
        molecule_mass /= 6.02214076e23
        molecule_volume = molecule_mass / target_density

        volume += molecule_volume * count

    volume = volume * unit.centimeter**3
    volume = volume.value_in_unit(unit.angstrom**3)
    return volume ** (1.0 / 3.0)


def _generate_input_file(
    components: list[tuple[str, int]],
    box_size: openmm.unit.Quantity,
    tolerance: openmm.unit.Quantity,
) -> str:
    """Generate the PACKMOL input file.

    Args:
        components: The list of components.
        box_size: The size of the box to pack the components into.
        tolerance: The PACKMOL convergence tolerance.

    Returns:
        The string contents of the PACKMOL input file.
    """

    box_size = box_size.value_in_unit(unit.angstrom)
    tolerance = tolerance.value_in_unit(unit.angstrom)

    logging.info(
        f"Generating PACKMOL coordinates with box size {box_size} Å and tolerance {tolerance}..."
    )

    seed = os.getenv("ABSOLV_PACKMOL_SEED")
    seed = seed if seed is not None else randint(1, 99999)

    return "\n".join(
        [
            f"tolerance {tolerance}",
            "filetype xyz",
            "output output.xyz",
            f"seed {seed}",
            "",
            *[
                f"structure {file_name}\n"
                f"  number {count}\n"
                f"  inside box 0. 0. 0. {box_size} {box_size} {box_size}\n"
                "end structure\n"
                ""
                for file_name, count in components
            ],
        ]
    )
