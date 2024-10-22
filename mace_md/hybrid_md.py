import sys
import MDAnalysis as mda
import MDAnalysis.transformations as tx
from ase.io import read
import mdtraj
from rdkit import Chem
import torch
import time
import numpy as np
from tempfile import mkstemp
from ase import Atoms
from rdkit.Chem.rdmolfiles import MolFromPDBFile, MolFromXYZFile
from openmm.openmm import Platform, System
from openmm import app
from typing import List, Tuple, Optional
from openmm import (
    LangevinMiddleIntegrator,
    # RPMDIntegrator,
    MonteCarloBarostat,
    CustomTorsionForce,
    # NoseHooverIntegrator,
    # VerletIntegrator,
)
import matplotlib.pyplot as plt
from mdtraj.geometry.dihedral import indices_phi, indices_psi
from openmm.app import (
    Simulation,
    StateDataReporter,
    PDBReporter,
    DCDReporter,
    CheckpointReporter,
    PDBFile,
    Modeller,
    CutoffNonPeriodic,
    PME,
    HBonds,
)
from ase.optimize import LBFGS
from openmm.app.metadynamics import Metadynamics, BiasVariable
from openmm.app.topology import Topology
from openmm.unit import (
    kelvin,
    picosecond,
    kilocalorie_per_mole,
    femtosecond,
    kilojoule_per_mole,
    picoseconds,
    femtoseconds,
    bar,
    nanometers,
    molar,
    angstrom,
)
from openff.toolkit.topology import Molecule
from openff.toolkit import ForceField

from openmmml import MLPotential

from mace_md.cli.utils import Solvents
from mace_md.enums import EXP_DENSITIES, SOLVENT_SMILES, ReplicaMixingScheme
from mace_md.nnp_repex.repex import (
    MixedSystemConstructor,
    RepexConstructor,
    get_atoms_from_resname,
)
from mace_md.nnp_repex.utils import (
    initialize_mm_forcefield,
    set_smff,
)
from tempfile import mkstemp
import os
import logging
from abc import ABC, abstractmethod

from mace_md.system_prep import (
    modeller_from_packmol,
    modeller_from_pdb,
    modeller_from_sdf,
    modeller_from_smiles,
    modeller_from_xyz,
    _approximate_num_molecules_by_density,
)


class MACESystemBase(ABC):
    temperature: float
    friction_coeff: float
    timestep: float
    dtype: torch.dtype
    output_dir: str
    openmm_precision: str
    SM_FF: str
    modeller: Modeller
    system: System
    mm_only: bool
    remove_cmm: bool
    unwrap: bool
    set_temperature: bool
    resname: str

    def __init__(
        self,
        file: str,
        model_path: str,
        output_dir: str,
        temperature: float,
        minimiser: str,
        resname: str,
        pressure: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
        timestep: float = 1.0,
        smff: str = "1.0",
        mm_only: bool = False,
        remove_cmm: bool = False,
        unwrap: bool = False,
        set_temperature: bool = False,
    ) -> None:
        super().__init__()

        self.file = file
        self.model_path = model_path
        self.temperature = temperature
        self.pressure = pressure
        self.timestep = timestep * femtosecond
        self.dtype = dtype
        self.set_temperature = set_temperature
        self.output_dir = output_dir
        self.remove_cmm = remove_cmm
        self.mm_only = mm_only
        self.minimiser = minimiser
        self.unwrap = unwrap
        self.resname = resname
        self.openmm_precision = "Double" if dtype == torch.float64 else "Mixed"
        logging.debug(f"OpenMM will use {self.openmm_precision} precision")

        self.SM_FF = set_smff(smff)
        logging.info(f"Using SMFF {self.SM_FF}")

        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_ase_atoms(self, file: str) -> Tuple[Atoms, Molecule]:
        """Generate the ase atoms object from the

        :param str file: file path or smiles
        :return Tuple[Atoms, Molecule]: ase Atoms object and initialised openFF molecule
        """
        # ml_mol can be a path to a file, or a smiles string
        if os.path.isfile(file):
            if file.endswith(".pdb"):
                # openFF refuses to work with pdb or xyz files, rely on rdkit to do the convertion to a mol first
                molecule = MolFromPDBFile(file)
                logging.warning(
                    "Initializing topology from pdb - this can lead to valence errors, check your starting structure carefully!"
                )
                molecule = Molecule.from_rdkit(
                    molecule, hydrogens_are_explicit=True, allow_undefined_stereo=True
                )
            elif file.endswith(".xyz"):
                molecule = MolFromXYZFile(file)
                molecule = Molecule.from_rdkit(molecule, hydrogens_are_explicit=True)
            else:
                # assume openFF will handle the format otherwise
                molecule = Molecule.from_file(file, allow_undefined_stereo=True)
        else:
            try:
                molecule = Molecule.from_smiles(file)
            except:
                raise ValueError(
                    f"Attempted to interpret arg {file} as a SMILES string, but could not parse"
                )

        _, tmpfile = mkstemp(suffix=".xyz")
        molecule._to_xyz_file(tmpfile)
        atoms = read(tmpfile)
        os.remove(tmpfile)
        return atoms, molecule

    @abstractmethod
    def create_system(self, **args):
        pass

    def propagate(
        self,
        steps: int,
        interval: int,
        restart: bool,
        run_metadynamics: bool = False,
        lambda_schedule: Optional[float] = None,
        platform: str = "CUDA",
    ):
        """Runs plain MD on the mixed system, writes a pdb trajectory

        :param int steps: number of steps to run the simulation for
        :param int interval: reportInterval attached to reporters
        """
        integrator = LangevinMiddleIntegrator(
            self.temperature, 1 / picoseconds, self.timestep
        )
        if self.pressure is not None:
            barostat = MonteCarloBarostat(self.pressure, self.temperature)
            self.system.addForce(barostat)

        if run_metadynamics:
            # if we have initialized from xyz, the topology won't have the information required to identify the cv indices, create from a pdb
            input_file = PDBFile(self.file)
            topology = input_file.getTopology()
            meta = self.run_metadynamics(
                topology=topology
                # cv1_dsl_string=self.cv1_dsl_string, cv2_dsl_string=self.cv2_dsl_string
            )
        # set alchemical state
        logging.debug(f"Running mixed MD for {steps} steps")
        simulation = Simulation(
            self.modeller.topology,
            self.system,
            integrator,
            platform=Platform.getPlatformByName(platform),
            platformProperties={"Precision": self.openmm_precision},
        )
        if lambda_schedule is not None:
            logging.info(f"Setting global lambda_interpolate to {lambda_schedule}")
            simulation.context.setParameter("lambda_interpolate", lambda_schedule)
        checkpoint_filepath = os.path.join(self.output_dir, "output.chk")
        if restart and os.path.isfile(checkpoint_filepath):
            with open(checkpoint_filepath, "rb") as f:
                logging.info("Loading simulation from checkpoint file...")
                simulation.context.loadCheckpoint(f.read())
        else:
            simulation.context.setPositions(self.modeller.getPositions())
            if self.minimiser == "openmm":
                logging.info("Minimising energy...")
                simulation.minimizeEnergy()
                minimised_state = simulation.context.getState(
                    getPositions=True, getVelocities=True, getForces=True
                )
                with open(
                    os.path.join(self.output_dir, "minimised_system.pdb"), "w"
                ) as f:
                    PDBFile.writeFile(
                        self.modeller.topology, minimised_state.getPositions(), file=f
                    )
            else:
                logging.info("Skipping minimisation step")

        if self.set_temperature:
            logging.info(f"Setting temperature to {self.temperature} K")
            simulation.context.setVelocitiesToTemperature(self.temperature)

        simulation.reporters.append(
            StateDataReporter(
                file=sys.stdout,
                reportInterval=interval,
                step=True,
                time=True,
                totalEnergy=True,
                potentialEnergy=True,
                density=True,
                volume=True,
                temperature=True,
                speed=True,
                progress=True,
                totalSteps=steps,
                remainingTime=True,
            )
        )
        simulation.reporters.append(
            StateDataReporter(
                file=os.path.join(self.output_dir, "mace_md.log"),
                reportInterval=interval,
                step=True,
                time=True,
                totalEnergy=True,
                potentialEnergy=True,
                density=True,
                volume=True,
                temperature=True,
                speed=True,
                progress=True,
                totalSteps=steps,
                remainingTime=True,
            )
        )
        simulation.reporters.append(
            PDBReporter(
                file=os.path.join(self.output_dir, "output.pdb"),
                reportInterval=interval,
                enforcePeriodicBox=False if self.unwrap else True,
            )
        )
        simulation.reporters.append(
            DCDReporter(
                file=os.path.join(self.output_dir, "output.dcd"),
                reportInterval=interval,
                append=(
                    restart
                    if os.path.isfile(os.path.join(self.output_dir, "output.dcd"))
                    else False
                ),
                enforcePeriodicBox=False if self.unwrap else True,
            )
        )

        # Add an extra hash to any existing checkpoint files
        checkpoint_files = [f for f in os.listdir(self.output_dir) if f.endswith("#")]
        for file in checkpoint_files:
            os.rename(
                os.path.join(self.output_dir, file),
                os.path.join(self.output_dir, f"{file}#"),
            )

        # backup the existing checkpoint file
        if os.path.isfile(checkpoint_filepath):
            os.rename(checkpoint_filepath, checkpoint_filepath + "#")
        checkpoint_reporter = CheckpointReporter(
            file=checkpoint_filepath, reportInterval=interval
        )
        simulation.reporters.append(checkpoint_reporter)

        if run_metadynamics:
            logging.info("Running metadynamics")
            # handles running the simulation with metadynamics
            meta.step(simulation, steps)

            fe = meta.getFreeEnergy()
            fig, ax = plt.subplots(1, 1)
            ax.imshow(fe)
            fig.savefig(os.path.join(self.output_dir, "free_energy.png"))
            # also write the numpy array to disk
            np.save(os.path.join(self.output_dir, "free_energy.npy"), fe)

        else:
            logging.info(f"Running dynamics for {steps} steps")
            simulation.step(steps)

        # write out centered strucure
        if steps > interval:
            self.write_centered_structure()

    def write_centered_structure(self):
        u = mda.Universe(
            os.path.join(self.output_dir, "prepared_system.pdb"),
            os.path.join(self.output_dir, "output.dcd"),
        )
        u.atoms.guess_bonds(vdwradii={"Cl": 1.81, "Br": 1.96})

        lig = u.select_atoms(f"resname {self.resname}")

        cell_dims = []
        for ts in u.trajectory:
            cell_dims.append(np.array(ts.dimensions))

        txs = [
            tx.unwrap(u.atoms),
            tx.center_in_box(lig),
            tx.wrap(u.atoms, compound="fragments"),
        ]

        u.trajectory.add_transformations(*txs)

        with mda.Writer(
            os.path.join(self.output_dir, "output_wrapped_equil.pdb"), u.atoms.n_atoms
        ) as W:
            # select last frame
            ts = u.trajectory[-1]
            W.write(u.atoms)

    def run_repex(
        self,
        replicas: int,
        restart: bool,
        steps: int,
        replica_mixing_scheme: ReplicaMixingScheme,
        equilibration_protocol: str,
        steps_per_mc_move: int = 1000,
        steps_per_equilibration_interval: int = 1000,
        lambda_schedule: Optional[List[float]] = None,
        checkpoint_interval: int = 100,
    ) -> None:
        repex_file_exists = os.path.isfile(os.path.join(self.output_dir, "repex.nc"))
        # even if restart has been set, disable if the checkpoint file was not found, enforce minimising the system
        if not repex_file_exists:
            restart = False

        sampler = RepexConstructor(
            mixed_system=self.system,
            initial_positions=self.modeller.positions,
            intervals_per_lambda_window=replicas,
            steps_per_equilibration_interval=steps_per_equilibration_interval,
            equilibration_protocol=equilibration_protocol,
            temperature=self.temperature * kelvin,
            lambda_schedule=lambda_schedule,
            n_states=replicas,
            restart=restart,
            mcmc_moves_kwargs={
                "timestep": 1.0 * femtoseconds,
                "collision_rate": 10.0 / picoseconds,
                "n_steps": steps_per_mc_move,
                "reassign_velocities": False,
                "n_restart_attempts": 20,
            },
            replica_exchange_sampler_kwargs={
                "number_of_iterations": steps,
                "online_analysis_interval": checkpoint_interval,
                "online_analysis_minimum_iterations": 10,
                "replica_mixing_scheme": replica_mixing_scheme,
            },
            storage_kwargs={
                "storage": os.path.join(self.output_dir, "repex.nc"),
                "checkpoint_interval": checkpoint_interval,
                "analysis_particle_indices": get_atoms_from_resname(
                    topology=self.modeller.topology,
                    nnpify_id=self.resname,
                    nnpify_type=self.nnpify_type,
                ),
            },
        ).sampler

        # do not minimsie if we are hot-starting the simulation from a checkpoint
        if not restart and equilibration_protocol == "minimise":
            logging.info("Minimizing system...")
            t1 = time.time()
            sampler.minimize()
            # just run a few steps to make sure the system is in a reasonable conformation

            logging.info(f"Minimised system  in {time.time() - t1} seconds")
            # we want to write out the positions after the minimisation - possibly something weird is going wrong here and it's ending up in a weird conformation

        sampler.run()

    def run_metadynamics(
        # self, topology: Topology, cv1_dsl_string: str, cv2_dsl_string: str
        self,
        topology: Topology,
    ) -> Metadynamics:
        # run well-tempered metadynamics
        mdtraj_topology = mdtraj.Topology.from_openmm(topology)

        cv1_atom_indices = indices_psi(mdtraj_topology)[1]
        cv2_atom_indices = indices_phi(mdtraj_topology)[1]
        logging.info(f"Selcted cv1 torsion atoms {cv1_atom_indices}")
        # logging.info(f"Selcted cv2 torsion atoms {cv2_atom_indices}")
        # takes the mixed system parametrised in the init method and performs metadynamics
        # in the canonical case, this should just use the psi-phi backbone angles of the peptide

        cv1 = CustomTorsionForce("theta")
        # cv1.addTorsion(cv1_atom_indices)
        cv1.addTorsion(*cv1_atom_indices)
        phi = BiasVariable(cv1, -np.pi, np.pi, biasWidth=0.5, periodic=True)

        cv2 = CustomTorsionForce("theta")
        cv2.addTorsion(*cv2_atom_indices)
        psi = BiasVariable(cv2, -np.pi, np.pi, biasWidth=0.5, periodic=True)
        os.makedirs(os.path.join(self.output_dir, "metaD"), exist_ok=True)
        meta = Metadynamics(
            self.system,
            [psi, phi],
            temperature=self.temperature,
            biasFactor=100.0,
            height=1.0 * kilojoule_per_mole,
            frequency=100,
            biasDir=os.path.join(self.output_dir, "metaD"),
            saveFrequency=100,
        )

        return meta


class MixedSystem(MACESystemBase):
    forcefields: List[str]
    padding: float
    ionicStrength: float
    nonbondedCutoff: float
    resname: str
    nnpify_type: str
    mixed_system: System
    minimise: bool
    water_model: str

    def __init__(
        self,
        file: str,
        ml_mol: str,
        model_path: str,
        resname: str,
        nnpify_type: str,
        minimiser: str,
        output_dir: str,
        padding: float = 1.2,
        shape: str = "cube",
        ionicStrength: float = 0.15,
        forcefields: List[str] = [
            "amber/protein.ff14SB.xml",
            "amber14/DNA.OL15.xml",
            "amber/tip3p_standard.xml",
        ],
        nonbondedCutoff: float = 1.0,
        temperature: float = 298,
        dtype: torch.dtype = torch.float64,
        decouple: bool = False,
        interpolate: bool = False,
        mm_only: bool = False,
        timestep: float = 1.0,
        smff: str = "1.0",
        water_model: str = "tip3p",
        pressure: Optional[float] = None,
        cv1: Optional[str] = None,
        cv2: Optional[str] = None,
        write_gmx: bool = False,
        unwrap=False,
        set_temperature=False,
    ) -> None:
        super().__init__(
            file=file,
            model_path=model_path,
            output_dir=output_dir,
            temperature=temperature,
            pressure=pressure,
            dtype=dtype,
            timestep=timestep,
            smff=smff,
            minimiser=minimiser,
            mm_only=mm_only,
            unwrap=unwrap,
            set_temperature=set_temperature,
            resname=resname,
        )

        self.forcefields = forcefields
        self.padding = padding
        self.shape = shape
        self.ionicStrength = ionicStrength
        self.nonbondedCutoff = nonbondedCutoff
        self.nnpify_type = nnpify_type
        self.cv1 = cv1
        self.cv2 = cv2
        self.water_model = water_model
        self.decouple = decouple
        self.interpolate = interpolate
        self.write_gmx = write_gmx

        logging.debug(f"OpenMM will use {self.openmm_precision} precision")

        # created the hybrid system
        self.create_system(
            file=file,
            ml_mol=ml_mol,
            model_path=model_path,
        )

    def create_system(
        self,
        file: str,
        model_path: str,
        ml_mol: str,
    ) -> None:
        """Creates the mixed system from a purely mm system

        :param str file: input pdb file
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        if ml_mol is not None:
            atoms, molecule = self.initialize_ase_atoms(ml_mol)
        else:
            atoms, molecule = None, None
        # set the default topology to that of the ml molecule, this will get overwritten below

        # Handle a complex, passed as a pdb file
        if file.endswith(".pdb"):
            input_file = PDBFile(file)
            topology = input_file.getTopology()

            self.modeller = Modeller(input_file.topology, input_file.positions)
            logging.info(
                f"Initialized topology with {len(input_file.positions)} positions"
            )

        # Handle a small molecule/small periodic system, passed as an sdf or xyz
        # this should also handle generating smirnoff parameters for something like an octa-acid, where this is still to be handled by the MM forcefield, but needs parameters generated
        elif file.endswith(".sdf") or file.endswith(".xyz"):
            # handle the case where the receptor and ligand are both passed as different sdf files:
            if ml_mol != file:
                logging.info("Combining and parametrising 2 sdf files...")
                receptor_as_molecule = Molecule.from_file(file)

                self.modeller = Modeller(
                    receptor_as_molecule.to_topology().to_openmm(),
                    get_xyz_from_mol(receptor_as_molecule.to_rdkit()) / 10,
                )
                ml_mol_modeller = Modeller(
                    molecule.to_topology().to_openmm(),
                    get_xyz_from_mol(molecule.to_rdkit()) / 10,
                )

                self.modeller.add(ml_mol_modeller.topology, ml_mol_modeller.positions)
                # send both to the forcefield initializer
                molecule = [molecule, receptor_as_molecule]

            else:
                input_file = molecule
                topology = molecule.to_topology().to_openmm()
                # Hold positions in nanometers
                positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

                logging.info(f"Initialized topology with {positions.shape} positions")

                self.modeller = Modeller(topology, positions)

        forcefield = initialize_mm_forcefield(
            molecule=molecule, forcefields=self.forcefields, smff=self.SM_FF
        )
        if self.write_gmx:
            from openff.interchange import Interchange

            interchange = Interchange.from_smirnoff(
                topology=molecule.to_topology(), force_field=ForceField(self.SM_FF)
            )
            interchange.to_top(os.path.join(self.output_dir, "topol.top"))
            interchange.to_gro(os.path.join(self.output_dir, "conf.gro"))
        if self.padding > 0:
            logging.info(f"Adding {self.shape} solvent box")
            if "tip4p" in self.water_model:
                self.modeller.addExtraParticles(forcefield)
            self.modeller.addSolvent(
                forcefield,
                model=self.water_model,
                padding=self.padding * nanometers,
                boxShape=self.shape,
                ionicStrength=self.ionicStrength * molar,
                neutralize=False,
            )

            omm_box_vecs = self.modeller.topology.getPeriodicBoxVectors()
            # ensure atoms object has boxvectors taken from the PDB file
            if atoms is not None:
                atoms.set_cell(
                    [
                        omm_box_vecs[0][0].value_in_unit(angstrom),
                        omm_box_vecs[1][1].value_in_unit(angstrom),
                        omm_box_vecs[2][2].value_in_unit(angstrom),
                    ]
                )
        system = forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod=(
                PME
                if self.modeller.topology.getPeriodicBoxVectors() is not None
                else CutoffNonPeriodic
            ),
            nonbondedCutoff=self.nonbondedCutoff * nanometers,
            constraints=None if "unconstrained" in self.SM_FF else HBonds,
        )

        # write the final prepared system to disk
        with open(os.path.join(self.output_dir, "prepared_system.pdb"), "w") as f:
            PDBFile.writeFile(
                self.modeller.topology, self.modeller.getPositions(), file=f
            )

        # if self.write_gmx:
        #     # write the openmm system to gromacs top/gro with parmed
        #     from parmed.openmm import load_topology

        #     parmed_structure = load_topology(self.modeller.topology, system)
        #     parmed_structure.save(os.path.join(self.output_dir, "topol_full.top"), overwrite=True)
        #     parmed_structure.save(os.path.join(self.output_dir, "conf_full.gro"), overwrite=True)
        #     raise KeyboardInterrupt

        if not self.decouple:
            if self.mm_only:
                logging.info("Creating MM system")
                self.system = system
            else:
                logging.debug("Creating hybrid system")
                self.system = MixedSystemConstructor(
                    system=system,
                    topology=self.modeller.topology,
                    nnpify_id=self.resname,
                    model_path=model_path,
                    nnp_potential=self.potential,
                    nnpify_type=self.nnpify_type,
                    atoms_obj=atoms,
                    interpolate=self.interpolate,
                    filename=model_path,
                    dtype=self.dtype,
                    max_n_pairs=self.max_n_pairs,
                ).mixed_system

            # optionally, add the alchemical customCVForce for the nonbonded interactions to run ABFE edges
        else:
            if not self.mm_only:
                # TODO: implement decoupled system for VdW/coulomb forces
                logging.info("Creating decoupled system")
                self.system = MixedSystemConstructor(
                    system=system,
                    topology=self.modeller.topology,
                    nnpify_type=self.nnpify_type,
                    nnpify_id=self.resname,
                    nnp_potential=self.potential,
                    model_path=model_path,
                    # cannot have the lambda parameter for this as well as the electrostatics/sterics being decoupled
                    interpolate=False,
                    atoms_obj=atoms,
                    filename=model_path,
                    dtype=self.dtype,
                ).mixed_system

            self.system = self.decouple_long_range(
                system,
                solute_indices=get_atoms_from_resname(
                    self.modeller.topology, self.resname, self.nnpify_type
                ),
            )


class PureSystem(MACESystemBase):
    potential: str
    temperature: float
    friction_coeff: float
    timestep: float
    dtype: torch.dtype
    output_dir: str
    openmm_precision: str
    modeller: Modeller
    boxsize: Optional[int]
    padding: float
    water_model: str
    box_shape: str

    def __init__(
        self,
        file: str,
        model_path: str,
        output_dir: str,
        temperature: float,
        minimiser: str,
        ligA_resname: str,
        ligB_resname: str,
        mcs_mapping: str,
        decouple: bool,
        pressure: Optional[float],
        dtype: torch.dtype,
        timestep: float,
        padding: float,
        box_shape: str,
        solvent: str = "tip3p",
        smff: str = "1.0",
        remove_cmm: bool = False,
        unwrap: bool = False,
        set_temperature: bool = False,
        nnpify_type: Optional[str] = None,
        optimized_model: bool = False,
        target_density: Optional[float] = None,
    ) -> None:
        super().__init__(
            file=file,
            model_path=model_path,
            output_dir=output_dir,
            temperature=temperature,
            pressure=pressure,
            dtype=dtype,
            timestep=timestep,
            smff=smff,
            resname=ligA_resname,
            minimiser=minimiser,
            remove_cmm=remove_cmm,
            unwrap=unwrap,
            set_temperature=set_temperature,
        )
        logging.debug(f"OpenMM will use {self.openmm_precision} precision")

        self.decouple = decouple
        self.nnpify_type = nnpify_type
        self.box_shape = box_shape
        self.solvent = solvent
        self.padding = padding
        self.optimized_model = optimized_model
        self.target_density = target_density
        self.ligA_resname = ligA_resname
        self.ligB_resname = ligB_resname
        self.mcs_mapping = []
        # parse a string of the form "0:1,1:2" into a list of 2-tuples
        if mcs_mapping is not None:
            for pair in mcs_mapping.split(","):
                self.mcs_mapping.append(tuple(map(int, pair.split(":"))))

        logging.debug(f"MCS mapping: {self.mcs_mapping}")

        self.create_system(file=file, model_path=model_path)

    def create_system(
        self,
        file: str,
        model_path: str,
    ) -> None:
        """Creates the openMM system with a TorchForce applied to all atoms in the system

        :param str file: input in pdb, xyz or smiles format
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        if self.solvent in ["tip3p", "tip4pew", None]:
            if file.endswith(".xyz"):
                self.modeller = modeller_from_xyz(file, self.padding, self.box_shape)
            elif file.endswith(".sdf"):
                self.modeller = modeller_from_sdf(file, self.padding, self.box_shape)
            elif file.endswith(".pdb"):
                self.modeller = modeller_from_pdb(file, self.padding, self.box_shape)
            elif Chem.MolFromSmiles(file) is not None:
                self.modeller = modeller_from_smiles(file, self.padding, self.box_shape)
        else:
            # treat solvent as a smiles string
            logging.info(f"Treating solvent {self.solvent} as a smiles string")
            # special case for solvating in non-aqueous media - use packmol
            # TODO: hardcoded solvent number for now, we should calculate the number required to fill a given box size
            # only 2 components for now
            n_solvent_molecules = _approximate_num_molecules_by_density(
                [self.solvent],
                padding=self.padding,
                target_density=self.target_density,
            )
            logging.info(f"Adding {n_solvent_molecules} solvent molecules")
            components = [(file, 1), (self.solvent, n_solvent_molecules)]
            self.modeller = modeller_from_packmol(
                components,
                box_target_density=self.target_density,
                # box_padding=box_padding,
            )

        logging.info(
            f"Initialized topology with {self.modeller.topology.getNumAtoms()} atoms"
        )

        with open(os.path.join(self.output_dir, "prepared_system.pdb"), "w") as f:
            PDBFile.writeFile(
                self.modeller.topology, self.modeller.getPositions(), file=f
            )

        ml_potential = MLPotential("mace", modelPath=model_path)
        if self.decouple:
            # solute_atoms = get_atoms_from_resname(
            #     self.modeller.topology, self.resname, self.nnpify_type
            # )
            logging.info(f"Creating alchemical system")
            self.system = ml_potential.createDualTopologySystem(
                self.modeller.topology,
                ligA_resname=self.ligA_resname,
                ligB_resname=self.ligB_resname,
                mcs_mapping=self.mcs_mapping,
                precision="single" if self.dtype == torch.float32 else "double",
                optimized_model=self.optimized_model,
            )
        else:
            self.system = ml_potential.createSystem(
                self.modeller.topology,
                dtype=self.dtype,
                max_n_pairs=-1,
                precision="single" if self.dtype == torch.float32 else "double",
                optimized_model=self.optimized_model,
            )

        if self.pressure is not None:
            logging.info(
                f"Pressure will be maintained at {self.pressure} bar with MC barostat"
            )
            barostat = MonteCarloBarostat(
                self.pressure * bar, self.temperature * kelvin
            )
            # barostat.setFrequency(25)  25 timestep is the default
            self.system.addForce(barostat)

        # if self.constrain_res is not None: assert len(self.constrain_res) == 2
        #     lig1_atoms = get_atoms_from_resname(
        #         self.modeller.topology, self.constrain_res[0], "resname"
        #     )
        #     lig2_atoms = get_atoms_from_resname(
        #         self.modeller.topology, self.constrain_res[1], "resname"
        #     )
        #     logging.info(f"Restraing atoms {lig1_atoms} and {lig2_atoms}")
        #     restraint = HarmonicRestraintForce(100, lig1_atoms, lig2_atoms)
        #     self.system.addForce(restraint)
