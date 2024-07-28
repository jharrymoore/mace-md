from typing import Optional, List, Iterable
from openmm import app
from openff.toolkit.topology import Molecule
from openmm.app import ForceField
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmmtools.alchemy import AlchemicalState
import openmm
from openmm.unit import kilojoules, mole, nanometer
import numpy as np
import logging
import copy


logger = logging.getLogger("mace_md")


def set_smff(smff: str) -> str:
    """Parse SMFF from command line and initialize the correct open-ff forcefield

    :param str smff: version of the OFF to use
    :raises ValueError: If SMFF version is not recognised
    :return str: the full filename for the OFF xml file
    """
    if smff == "1.0":
        return "openff_unconstrained-1.0.0.offxml"
    elif smff == "2.0":
        return "openff_unconstrained-2.0.0.offxml"
    elif smff == "2.0-constrained":
        return "openff-2.0.0.offxml"
    else:
        raise ValueError(f"Small molecule forcefield {smff} not recognised")


def initialize_mm_forcefield(
    molecule: Optional[Molecule],
    forcefields: List = ["amber/protein.ff14SB.xml"],
    smff: str = "openff_unconstrained-1.0.0.offxml",
) -> ForceField:
    forcefield = ForceField(*forcefields)
    if molecule is not None:
        if isinstance(molecule, Molecule):
            molecule = [molecule]
        for mol in molecule:
            logger.info(f"Adding {mol} to forcefield")
            smirnoff = SMIRNOFFTemplateGenerator(molecules=mol, forcefield=smff)
            forcefield.registerTemplateGenerator(smirnoff.generator)
    return forcefield


# taken from peastman/openmm-ml
def remove_bonded_forces(
    system: openmm.System,
    atoms: Iterable[int],
    removeInSet: bool,
    removeConstraints: bool,
) -> openmm.System:
    """Copy a System, removing all bonded interactions between atoms in (or not in) a particular set.

    Parameters
    ----------
    system: System
        the System to copy
    atoms: Iterable[int]
        a set of atom indices
    removeInSet: bool
        if True, any bonded term connecting atoms in the specified set is removed.  If False,
        any term that does *not* connect atoms in the specified set is removed
    removeConstraints: bool
        if True, remove constraints between pairs of atoms in the set

    Returns
    -------
    a newly created System object in which the specified bonded interactions have been removed
    """
    atomSet = set(atoms)

    # Create an XML representation of the System.

    import xml.etree.ElementTree as ET

    xml = openmm.XmlSerializer.serialize(system)
    root = ET.fromstring(xml)

    # This function decides whether a bonded interaction should be removed.

    def shouldRemove(termAtoms):
        return all(a in atomSet for a in termAtoms) == removeInSet

    # Remove bonds, angles, and torsions.

    for bonds in root.findall("./Forces/Force/Bonds"):
        for bond in bonds.findall("Bond"):
            bondAtoms = [int(bond.attrib[p]) for p in ("p1", "p2")]
            if shouldRemove(bondAtoms):
                bonds.remove(bond)
    for angles in root.findall("./Forces/Force/Angles"):
        for angle in angles.findall("Angle"):
            angleAtoms = [int(angle.attrib[p]) for p in ("p1", "p2", "p3")]
            if shouldRemove(angleAtoms):
                angles.remove(angle)
    for torsions in root.findall("./Forces/Force/Torsions"):
        for torsion in torsions.findall("Torsion"):
            torsionAtoms = [int(torsion.attrib[p]) for p in ("p1", "p2", "p3", "p4")]
            if shouldRemove(torsionAtoms):
                torsions.remove(torsion)

    # Optionally remove constraints.

    if removeConstraints:
        for constraints in root.findall("./Constraints"):
            for constraint in constraints.findall("Constraint"):
                constraintAtoms = [int(constraint.attrib[p]) for p in ("p1", "p2")]
                if shouldRemove(constraintAtoms):
                    constraints.remove(constraint)

    # Create a new System from it.

    return openmm.XmlSerializer.deserialize(ET.tostring(root, encoding="unicode"))


class NNPProtocol:
    """
    protocol for perturbing the `lambda_interpolate` parameter of an openmm-ml mixed system
    """

    default_functions = {"lambda_interpolate": lambda x: x}

    def __init__(self, temp_scale=None, **unused_kwargs):

        """allow to encode a temp scaling"""
        if temp_scale is not None:  # if the temp scale is not none, it must be a float
            assert type(temp_scale) == float, f"temp scale is not a float"

            def interREST_fn(x):
                if x <= 0.5:
                    out = -2.0 * x * (1.0 - temp_scale)
                else:
                    out = 2 * x * (1 - temp_scale) + 2 * temp_scale - 2
                print("Output of interREST_fn is: ", out, x, temp_scale)
                return out

            self.functions = copy.deepcopy(self.default_functions)
            # set the modified function as partialed with the `temp_scale`; need to check that this is right.
            self.functions["lambda_interRest"] = interREST_fn
        else:  # remove the temp scale
            my_default_fns = copy.deepcopy(self.default_functions)
            out_default_functions = {
                "lambda_interpolate": my_default_fns["lambda_interpolate"]
            }
            self.functions = out_default_functions


class NNPAlchemicalState(AlchemicalState):
    """
    neural network potential flavor of `AlchemicalState` for perturbing the `lambda_interpolate` value
    """

    class _LambdaParameter(AlchemicalState._LambdaParameter):
        pass

    lambda_interpolate = _LambdaParameter("lambda_interpolate")
    lambda_interREST = _LambdaParameter(
        "lambda_interREST"
    )  # this is specific to the inter-REST region

    def set_alchemical_parameters(
        self, global_lambda, lambda_protocol=NNPProtocol(1.5), **unused_kwargs
    ):
        self.global_lambda = global_lambda
        for parameter_name in lambda_protocol.functions:
            lambda_value = lambda_protocol.functions[parameter_name](global_lambda)
            setattr(self, parameter_name, lambda_value)

:

class NNPCompatibilityMixin(object):
    """
    Mixin for subclasses of `MultistateSampler` that supports `openmm-ml` exchanges of `lambda_interpolate`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(
        self,
        n_states,
        mixed_system,
        init_positions,
        temperature,
        storage_kwargs,
        equilibration_protocol: str,
        replica_exchange_sampler_kwargs: dict,
        topology: app.Topology,
        system_id: str,
        n_replicas=None,
        lambda_schedule: Optional[Iterable[float]] = None,
        lambda_protocol=None,
        setup_equilibration_intervals=None,
        steps_per_setup_equilibration_interval=None,
        **unused_kwargs,
    ):
        """try to gently equilibrate the setup of the different thermodynamic states;
        make the number of `setup_equilibration_intervals` some multiple of `n_states`.
        The number of initial equilibration steps will be equal to
        `setup_equilibration_intervals * steps_per_setup_equilibration_interval`
        """
        import openmm
        from openmmtools.states import (
            ThermodynamicState,
            SamplerState,
            CompoundThermodynamicState,
        )
        from copy import deepcopy
        from openmmtools.multistate import MultiStateReporter
        from openmmtools.utils import get_fastest_platform
        from openmmtools import cache

        platform = get_fastest_platform(minimum_precision="mixed")

        if lambda_schedule is None:
            lambda_schedule = np.linspace(0.0, 1.0, n_states)
        logger.info(f"Using lambda schedule {lambda_schedule}")

        platform = get_fastest_platform(minimum_precision="mixed")
        context_cache = cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform
        )

        if equilibration_protocol not in ["minimise", "gentle"]:
            raise ValueError(
                f"equilibration protocol {equilibration_protocol} not recognised"
            )

        logger.info(f"Using {equilibration_protocol} equilibration protocol")
        lambda_zero_alchemical_state = NNPAlchemicalState.from_system(mixed_system)
        thermostate = ThermodynamicState(mixed_system, temperature=temperature)
        compound_thermostate = CompoundThermodynamicState(
            thermostate, composable_states=[lambda_zero_alchemical_state]
        )
        thermostate_list, sampler_state_list, unsampled_thermostate_list = [], [], []
        if n_replicas is None:
            n_replicas = n_states
        else:
            raise NotImplementedError(
                f"""the number of states was given as {n_states} 
                                        but the number of replicas was given as {n_replicas}. 
                                        We currently only support equal states and replicas"""
            )

        if lambda_protocol is None:

            lambda_protocol = NNPProtocol()
        else:
            raise NotImplementedError(
                f"""`lambda_protocol` is currently placeholding; only default `None` 
                                      is allowed until the `lambda_protocol` class is appropriately generalized"""
            )
        init_sampler_state = SamplerState(
            init_positions, box_vectors=mixed_system.getDefaultPeriodicBoxVectors()
        )
        if equilibration_protocol == "gentle":
            if setup_equilibration_intervals is not None:
                # attempt to gently equilibrate
                assert (
                    setup_equilibration_intervals % n_states == 0
                ), f"""
                the number of `n_states` must be divisible into `setup_equilibration_intervals`"""
                interval_stepper = setup_equilibration_intervals // n_states

            else:
                raise Exception(
                    f"At present, we require setup equilibration interval work."
                )
            # first, a context, integrator to equilibrate and minimize state 0
            eq_context, eq_integrator = context_cache.get_context(
                deepcopy(compound_thermostate),
                openmm.LangevinMiddleIntegrator(temperature, 1.0, 0.001),
            )

            eq_context.setParameter("lambda_interpolate", 0.0)
            init_sampler_state.apply_to_context(eq_context)
            logger.info("Minimising initial state")
            openmm.LocalEnergyMinimizer.minimize(eq_context)  # don't forget to minimize
            # update from context for good measure
            init_sampler_state.update_from_context(eq_context)

            logger.info(f"making lambda states...")
            lambda_subinterval_schedule = np.linspace(
                0.0, 1.0, setup_equilibration_intervals
            )
            # now compute the indices of the subinterval schedule that will correspond to a state in the lambda schedule
            subinterval_matching_idx = np.round(
                np.linspace(0, lambda_subinterval_schedule.shape[0] - 1, n_states)
            ).astype(int)

            for idx, lambda_subinterval in enumerate(lambda_subinterval_schedule):
                logger.info(f"running lambda subinterval {lambda_subinterval}.")
                # copy thermostate
                compound_thermostate_copy = deepcopy(compound_thermostate)
                # update thermostate
                compound_thermostate_copy.set_alchemical_parameters(
                    lambda_subinterval, lambda_protocol
                )

                compound_thermostate_copy.apply_to_context(eq_context)
                # step the integrator
                eq_integrator.step(steps_per_setup_equilibration_interval)

                if idx in subinterval_matching_idx:
                    thermostate_list.append(compound_thermostate_copy)
                    sampler_state_list.append(deepcopy(init_sampler_state))

            # put context, integrator into garbage collector
            del eq_context
            del eq_integrator

        elif equilibration_protocol == "minimise":
            for lambda_val in lambda_schedule:
                compound_thermostate_copy = deepcopy(compound_thermostate)
                compound_thermostate_copy.set_alchemical_parameters(
                    lambda_val, lambda_protocol
                )
                thermostate_list.append(compound_thermostate_copy)
                sampler_state_list.append(deepcopy(init_sampler_state))

        reporter = MultiStateReporter(**storage_kwargs)
        elements = [a.element for a in topology.get_atoms()]
        self.create(
            thermodynamic_states=thermostate_list,
            sampler_states=sampler_state_list,
            storage=reporter,
            # store lambda schedule, steps per iteration, swap_protocol, atom types
        
            metadata={
                "md_steps_per_iter": replica_exchange_sampler_kwargs["steps_per_iteration"],
                "lambda_schedule": np.array(lambda_schedule),
                "replica_mixing_scheme": replica_exchange_sampler_kwargs["replica_mixing_scheme"],
                "elements":elements,
                "system_id": system_id


            }
        )

