# unit tests for mace-md entrypoints
# uses pytest to run
import torch
import intel_extension_for_pytorch as ipex
import openmmtorch
import openmm
import os
import pytest
import shutil
from openmmtools.integrators import (
    LangevinIntegrator,
    NoseHooverChainVelocityVerletIntegrator,
)

from mace_md.hybrid_md import PureSystem, MixedSystem
from openmm import unit
import logging
import tempfile


print(openmm)

torch.set_default_dtype(torch.float64)

TEST_DIR = "examples/example_data"
JUNK_DIR = tempfile.mkdtemp()
model_path = os.path.join(TEST_DIR, "MACE_test.model")
# model_path = os.path.join(TEST_DIR, "14-mptrj-slower-lr-13.model")
logger = logging.getLogger("DEBUG")


@pytest.mark.parametrize("remove_cmm", [True, False])
@pytest.mark.parametrize("file", ["ejm_31.sdf", "waterbox.xyz"])
@pytest.mark.parametrize("integrator", ["langevin", "nose-hoover", "verlet"])
@pytest.mark.parametrize("nl", ["torch", "nnpops"])
@pytest.mark.parametrize("minimiser", ["ase", "openmm"])
@pytest.mark.parametrize("platform", ["OpenCL"])
def test_pure_mace_md(file, nl, remove_cmm, minimiser, integrator, platform):
    file_stub = file.split(".")[0]
    cmm = "cmm" if remove_cmm else "nocmm"

    # inspect the dtype of the model
    model = torch.load(model_path)
    dtype = model.r_max.dtype


    if minimiser == "ase" and torch.cuda.device_count() < 1:
        pytest.skip("ASE requires a CUDA device")


    system = PureSystem(
        ml_mol=os.path.join(TEST_DIR, file),
        model_path=model_path,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl=nl,
        dtype=dtype,
        pressure=1.0 if file_stub == "waterbox" else None,
        remove_cmm=remove_cmm,
        minimiser=minimiser,
        max_n_pairs=-1,
    )
    output_file = f"output_pure_{file_stub}_{nl}_{cmm}.pdb"
    system.run_mixed_md(
        steps=20,
        interval=5,
        output_file=output_file,
        restart=False,
        integrator_name=integrator,
        platform=platform,
    )

    # check the output file exists and is larger than 0 bytes
    assert os.path.exists(os.path.join(JUNK_DIR, output_file))
    assert os.path.getsize(os.path.join(JUNK_DIR, output_file)) > 0


@pytest.mark.parametrize("water_model", ["tip3p", "tip4pew"])
@pytest.mark.parametrize("mm_only", [True, False])
@pytest.mark.parametrize("remove_cmm", [True, False])
@pytest.mark.parametrize("nl", ["torch", "nnpops"])
@pytest.mark.parametrize("integrator", ["langevin", "nose-hoover", "verlet"])
@pytest.mark.parametrize("minimiser", ["openmm", None])
def test_hybrid_system_md(nl, remove_cmm, mm_only, water_model, minimiser, integrator):
    cmm = "cmm" if remove_cmm else "nocmm"
    mm = "mm_only" if mm_only else "mm_and_ml"
    forcefields = (
        ["amber/protein.ff14SB.xml", "amber14/DNA.OL15.xml", "amber/tip3p_standard.xml"]
        if water_model == "tip3p"
        else ["amber/protein.ff14SB.xml", "amber14/DNA.OL15.xml", "amber14/tip4pew.xml"]
    )
    system = MixedSystem(
        file=os.path.join(TEST_DIR, "ejm_31.sdf"),
        ml_mol=os.path.join(TEST_DIR, "ejm_31.sdf"),
        model_path=model_path,
        forcefields=forcefields,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl=nl,
        nnpify_type="resname",
        resname="UNK",
        minimiser=minimiser,
        water_model=water_model,
        max_n_pairs=-1,
    )
    output_file = f"output_hybrid_{nl}_{cmm}_{mm}.pdb"

    system.run_mixed_md(
        steps=20,
        interval=5,
        output_file=output_file,
        restart=False,
        integrator_name=integrator,
    )
    assert os.path.exists(os.path.join(JUNK_DIR, output_file))
    assert os.path.getsize(os.path.join(JUNK_DIR, output_file)) > 0


@pytest.mark.parametrize("remove_cmm", [True, False])
@pytest.mark.parametrize("nl", ["torch", "nnpops"])
@pytest.mark.parametrize("minimiser", ["ase", None])
def test_rpmd(nl, remove_cmm, minimiser):
    cmm = "cmm" if remove_cmm else "nocmm"
    cmm = "cmm" if remove_cmm else "nocmm"
    system = PureSystem(
        ml_mol=os.path.join(TEST_DIR, "ejm_31.sdf"),
        model_path=model_path,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl=nl,
        pressure=None,
        remove_cmm=remove_cmm,
        minimiser=minimiser,
        max_n_pairs=-1,
    )
    output_file = f"output_hybrid_rpmd_{nl}_{cmm}.pdb"

    system.run_mixed_md(
        steps=20,
        interval=5,
        output_file=output_file,
        restart=False,
        integrator_name="rpmd",
    )

    assert os.path.exists(os.path.join(JUNK_DIR, output_file))
    assert os.path.getsize(os.path.join(JUNK_DIR, output_file)) > 0


# mark as slow test
@pytest.mark.slow
@pytest.mark.parametrize("water_model", ["tip3p"])
@pytest.mark.parametrize("remove_cmm", [False])
@pytest.mark.parametrize("nl", ["nnpops"])
@pytest.mark.parametrize("minimiser", ["openmm"])
def test_hybrid_system_repex(nl, remove_cmm, water_model, minimiser):
    cmm = "cmm" if remove_cmm else "nocmm"
    forcefields = (
        ["amber/protein.ff14SB.xml", "amber14/DNA.OL15.xml", "amber/tip3p_standard.xml"]
        if water_model == "tip3p"
        else ["amber/protein.ff14SB.xml", "amber14/DNA.OL15.xml", "amber14/tip4pew.xml"]
    )
    system = MixedSystem(
        file=os.path.join(TEST_DIR, "ejm_31.sdf"),
        ml_mol=os.path.join(TEST_DIR, "ejm_31.sdf"),
        model_path=model_path,
        forcefields=forcefields,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl=nl,
        nnpify_type="resname",
        interpolate=True,
        resname="UNK",
        minimiser=minimiser,
        water_model=water_model,
        max_n_pairs=-1,
    )

    system.run_repex(
        replicas=2,
        steps=3,
        steps_per_mc_move=100,
        checkpoint_interval=1,
        decouple=False,
        restart=False,
    )
    assert os.path.exists(os.path.join(JUNK_DIR, "repex.nc"))
    assert os.path.getsize(os.path.join(JUNK_DIR, "repex.nc")) > 0


# @pytest.mark.parametrize("remove_cmm", [True, False])
# @pytest.mark.parametrize("nl", ["torch", "nnpops"])
# @pytest.mark.parametrize("minimiser", ["openmm", None])
# @pytest.mark.slow
def test_pure_system_repex():

    shutil.rmtree(os.path.join(JUNK_DIR, "repex_pure"))
    os.makedirs(os.path.join(JUNK_DIR, "repex_pure"))

    system = PureSystem(
        file=os.path.join(TEST_DIR, "ejm31_solvated.pdb"),
        ml_mol=os.path.join(TEST_DIR, "ejm31_solvated.pdb"),
        model_path=model_path,
        potential="mace",
        output_dir=os.path.join(JUNK_DIR, "repex_pure"),
        temperature=298,
        nl="nnpops",
        pressure=None,
        remove_cmm=False,
        minimiser="openmm",
        decouple=True,
        max_n_pairs=-1,
        resname="UNK",
        nnpify_type="resname",
    )

    system.run_repex(
        replicas=2,
        steps=3,
        steps_per_mc_move=100,
        checkpoint_interval=1,
        decouple=False,
        restart=False,
    )
    assert os.path.exists(os.path.join(JUNK_DIR, "repex.nc"))
    assert os.path.getsize(os.path.join(JUNK_DIR, "repex.nc")) > 0
