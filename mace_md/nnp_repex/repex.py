import logging

import os
from openmmml.mlpotential import MLPotential
import openmm
from openmm import unit, app
from openmm.app import Topology
import mdtraj
from openmmtools import mcmc
from openmmtools.multistate import replicaexchange, multistatesampler
from mace_md.nnp_repex import NNPCompatibilityMixin
from typing import Dict, Iterable, Optional, List
import os
import logging

from openmmtools.states import SamplerState


def deserialize_xml(filename):
    with open(filename, "r") as infile:
        xml_readable = infile.read()
    xml_deserialized = openmm.XmlSerializer.deserialize(xml_readable)
    return xml_deserialized


class NNPRepexSampler(NNPCompatibilityMixin, replicaexchange.ReplicaExchangeSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NNPMultiStateSampler(NNPCompatibilityMixin, multistatesampler.MultiStateSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_atoms_from_resname(
    topology: Topology, nnpify_id: str, nnpify_type: str
) -> List:
    """get the atoms (in order) of the appropriate topology resname"""
    if nnpify_type == "chain":
        topology = mdtraj.Topology.from_openmm(topology)
        atoms = topology.select(f"chainid == {nnpify_id}")
        return atoms
    elif nnpify_type == "resname":
        all_resnames = [
            res.name for res in topology.residues() if res.name == nnpify_id
        ]
        assert (
            len(all_resnames) == 1
        ), f"did not find exactly 1 residue with the name {nnpify_id}; found {len(all_resnames)}"
        for residue in list(topology.residues()):
            if residue.name == nnpify_id:
                break
        atoms = []
        for atom in list(residue.atoms()):
            atoms.append(atom.index)
        assert (
            sorted(atoms) == atoms
        ), f"atom indices ({atoms}) are not in ascending order"
        return atoms
    else:
        raise ValueError("Either chain or resname must be set")


def assert_no_residue_constraints(system: openmm.System, atoms: Iterable[int]):
    """
    assert that there are no constraints within the nnp region before making mixed system
    """
    atom_set = set(atoms)
    all_constraints = []
    for idx in range(system.getNumConstraints()):
        p1, p2, _ = system.getConstraintParameters(idx)
        all_constraints.append(p1)
        all_constraints.append(p2)
    set_intersect = set(all_constraints) & atom_set
    if set_intersect:
        raise Exception(
            f"the intersection of system constraints and the specified atom set is not empty: {set_intersect}"
        )


class MixedSystemConstructor:
    """simple handler to make vanilla `openmm.System` objects a mixedSystem with an `openmm.TorchForce`"""

    def __init__(
        self,
        system: openmm.System,
        topology: app.topology.Topology,
        nnpify_id: str,
        model_path: str,
        nnpify_type: str = "resname",
        nnp_potential: str = "ani2x",
        implementation: str = "nnpops",
        interpolate: bool = True,
        atom_indices=None,
        **createMixedSystem_kwargs,
    ):
        """
        initialize the constructor
        """
        self._system = system
        self._topology = topology
        self._nnpify_id = nnpify_id
        self._implementation = implementation
        self._interpolate = interpolate

        self._atom_indices = (
            get_atoms_from_resname(topology, nnpify_id, nnpify_type)
            if atom_indices is None
            else atom_indices
        )
        print(f"Treating atom indices {self._atom_indices} with ML potential")
        assert_no_residue_constraints(system, self._atom_indices)
        self._nnp_potential_str = nnp_potential
        self._nnp_potential = MLPotential(
            self._nnp_potential_str, model_path=model_path
        )
        # pop the atoms_obj from the kwargs
        self._createMixedSystem_kwargs = createMixedSystem_kwargs

    @property
    def mixed_system(self):
        return self._nnp_potential.createMixedSystem(
            self._topology,
            system=self._system,
            atoms=self._atom_indices,
            implementation="nnpops",
            interpolate=self._interpolate,
            **self._createMixedSystem_kwargs,
        )


class RepexConstructor:
    """
    simple handler to build replica exchange sampler.
    """

    def __init__(
        self,
        mixed_system: openmm.System,
        initial_positions: unit.Quantity,
        n_states: int,
        temperature: unit.Quantity,
        intervals_per_lambda_window: int,
        steps_per_equilibration_interval: int,
        equilibration_protocol: str,
        mcmc_moves_kwargs: Dict,
        topology: app.Topology,
        storage_kwargs,
        replica_exchange_sampler_kwargs,
        lambda_schedule: Optional[Iterable[float]] = None,
        restart: bool = False,
        mcmc_moves: Optional[
            mcmc.MCMCMove
        ] = mcmc.LangevinDynamicsMove,  # MiddleIntegrator
        **kwargs,
    ):
        self._mixed_system = mixed_system
        self._storage_kwargs = storage_kwargs
        self._temperature = temperature
        self._mcmc_moves = mcmc_moves
        self._mcmc_moves_kwargs = mcmc_moves_kwargs
        self._replica_exchange_sampler_kwargs = replica_exchange_sampler_kwargs
        self._n_states = n_states
        self.restart = restart
        self._intervals_per_lambda_window = intervals_per_lambda_window
        self._steps_per_equilibration_interval = steps_per_equilibration_interval
        self._equilibration_protocol = equilibration_protocol
        self._extra_kwargs = kwargs
        self._lambda_schedule = lambda_schedule
        self._topology = topology

        # initial positions
        self._initial_positions = initial_positions

    @property
    def sampler(self):
        # set context cache
        from openmmtools.utils import get_fastest_platform
        from openmmtools import cache

        platform = get_fastest_platform(minimum_precision="mixed")
        context_cache = cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform
        )
        mcmc_moves = self._mcmc_moves(**self._mcmc_moves_kwargs)
        if os.path.isfile(self._storage_kwargs["storage"]) and not self.restart:
            # file exists and restart not requested, remove the storage file before continuing
            logging.info(f"Removing storage file {self._storage_kwargs['storage']}")
            os.remove(self._storage_kwargs["storage"])
        if os.path.isfile(self._storage_kwargs["storage"]) and self.restart:
            # repex.nc file exists, attempt to restart from this file
            logging.info(
                f"Restarting simulation from file {self._storage_kwargs['storage']}"
            )
            _sampler = NNPRepexSampler.from_storage(self._storage_kwargs["storage"])
            _sampler.energy_context_cache = context_cache
            _sampler.sampler_context_cache = context_cache
            _sampler.number_of_iterations = self._replica_exchange_sampler_kwargs[
                "number_of_iterations"
            ]
            _sampler.replica_mixing_scheme = self._replica_exchange_sampler_kwargs[
                "replica_mixing_scheme"
            ]
        else:
            logging.info("Starting Repex sampling from scratch")
            _sampler = NNPRepexSampler(
                mcmc_moves=mcmc_moves, **self._replica_exchange_sampler_kwargs
            )
            _sampler.energy_context_cache = context_cache
            _sampler.sampler_context_cache = context_cache
            _sampler.setup(
                n_states=self._n_states,
                mixed_system=self._mixed_system,
                init_positions=self._initial_positions,
                temperature=self._temperature,
                lambda_schedule=self._lambda_schedule,
                storage_kwargs=self._storage_kwargs,
                setup_equilibration_intervals=self._intervals_per_lambda_window,
                equilibration_protocol=self._equilibration_protocol,
                steps_per_setup_equilibration_interval=self._steps_per_equilibration_interval,
                replica_exchange_sampler_kwargs=self._replica_exchange_sampler_kwargs,
                topology=self._topology,
                **self._extra_kwargs,
            )
        return _sampler
