from mace_md.cli.utils import RunType
from mace_md.hybrid_md import PureSystem, MixedSystem
import logging
from mace_md.cli.utils import parse_arguments
import torch
import os
from prettytable import PrettyTable
import time
import ast

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

torch._C._jit_set_nvfuser_enabled(False)


class ConsoleColours:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    ORANGE = "\033[0;33m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BLINKING = "\33[5m"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-[%(levelname)s]-%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("mace_md.log")],
)


def main():
    banner = f"""{ConsoleColours.CYAN}



====================================================================================================
ooo        ooooo       .o.         .oooooo.   oooooooooooo         ooo        ooooo oooooooooo.   
`88.       .888'      .888.       d8P'  `Y8b  `888'     `8         `88.       .888' `888'   `Y8b  
 888b     d'888      .8"888.     888           888                  888b     d'888   888      888 
 8 Y88. .P  888     .8' `888.    888           888oooo8             8 Y88. .P  888   888      888 
 8  `888'   888    .88ooo8888.   888           888    "    8888888  8  `888'   888   888      888 
 8    Y     888   .8'     `888.  `88b    ooo   888       o          8    Y     888   888     d88' 
o8o        o888o o88o     o8888o  `Y8bood8P'  o888ooooood8         o8o        o888o o888bood8P'   
====================================================================================================
{ConsoleColours.ENDC}                                                                                                  
                                                                                                  
                                                                                                  
"""
    try:
        width = os.get_terminal_size().columns
    except (OSError, AttributeError):
        width = 150
    for line in banner.split("\n"):
        print(line.center(width))

    args = parse_arguments().parse_args()
    if args.lambda_schedule is not None:
        args.lambda_schedule = ast.literal_eval(args.lambda_schedule)
        print(args.lambda_schedule)
        if args.replicas is not None:
            if args.replicas != len(args.lambda_schedule):
                logging.warning(
                    "Number of replicas does not match the length of the lambda schedule. '--replicas' argument will be ignored"
                )
    x = PrettyTable()
    x.field_names = ["Argument", "Value"]
    for arg in vars(args):
        x.add_row([arg, getattr(args, arg)])
    print(x)

    if args.dtype == "float32":
        logging.info("Running with single precision ")
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32
    elif args.dtype == "float64":
        logging.info("Running with double precision")
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
    else:
        raise ValueError(f"Data type {args.dtype} not recognised")
    # tools.setup_logger(level=args.log_level, directory=args.output_dir, tag="mace_md")

    # we don't need to specify the file twice if dealing with just the ligand
    if args.ml_mol is None:
        args.ml_mol = args.file
    if args.solvent == "tip3p":
        args.forcefields.append("amber/tip3p_standard.xml")
    elif args.solvent == "tip4pew":
        args.forcefields.append("amber14/tip4pew.xml")
    # else:
    #     raise ValueError(f"Water model {args.solvent} not recognised")

    if args.mm_only and args.system_type == "pure":
        raise ValueError(
            "Cannot run a pure MACE system with only the MM forcefield\
                 - please use a hybrid system"
        )
    if args.constrain_res is not None:
        args.constrain_res = ast.literal_eval(args.constrain_atoms)
        assert isinstance(args.constrain_res, list), "constrain_atoms must be a list"
    # Only need interpolation when running repex and not decoupling
    interpolate = (
        True if (args.run_type in ["repex", "neq"] and not args.decouple) else False
    )

    if args.minimiser == "ase" and args.system_type != "pure":
        raise ValueError("Cannot use ASE minimiser with a hybrid system, use openmm")

    if args.system_type == "pure":
        # if we're running a pure system, we need to specify the ml_mol,
        # args.file is only useful for metadynamics where we need the
        # topology to extract the right CV atoms
        system = PureSystem(
            file=args.file,
            # ml_mol=args.ml_mol,
            model_path=args.model_path,
            output_dir=args.output_dir,
            temperature=args.temperature,
            pressure=args.pressure,
            dtype=dtype,
            decouple=args.decouple,
            timestep=args.timestep,
            smff=args.smff,
            minimiser=args.minimiser,
            padding=args.padding,
            box_shape=args.box_shape,
            unwrap=args.unwrap,
            solvent=args.solvent,
            set_temperature=args.set_temperature,
            resname=args.resname,
            nnpify_type=args.ml_selection,
            optimized_model=args.optimized_model,
            target_density=args.target_density,
        )

    elif args.system_type == "hybrid":
        system = MixedSystem(
            file=args.file,
            ml_mol=args.ml_mol,
            model_path=args.model_path,
            forcefields=args.forcefields,
            resname=args.resname,
            nnpify_type=args.ml_selection,
            ionicStrength=args.ionic_strength,
            nonbondedCutoff=args.nonbondedCutoff,
            padding=args.padding,
            shape=args.box_shape,
            temperature=args.temperature,
            dtype=dtype,
            output_dir=args.output_dir,
            smff=args.smff,
            pressure=args.pressure,
            decouple=args.decouple,
            interpolate=interpolate,
            minimiser=args.minimiser,
            mm_only=args.mm_only,
            water_model=args.solvent,
            write_gmx=args.write_gmx,
            unwrap=args.unwrap,
            set_temperature=args.set_temperature,
        )
    else:
        raise ValueError(f"System type {args.system_type} not recognised!")
    if args.run_type == RunType.MD:
        # if we are running regular MD, but with decouple, we can effectively run a single replica - set the system's alchemical parameter
        # set the lambda value to the specified value
        if args.lambda_schedule is not None:
            assert (
                len(args.lambda_schedule) == 1
            ), "If running regular MD with decouple, only one lambda value should be specified"
            args.lambda_schedule = args.lambda_schedule[0]

        system.propagate(
            steps=args.steps,
            interval=args.interval,
            run_metadynamics=args.meta,
            lambda_schedule=args.lambda_schedule,
            restart=args.restart,
        )
    elif args.run_type == RunType.REPEX:
        system.run_repex(
            replicas=args.replicas,
            restart=args.restart,
            steps=args.steps,
            lambda_schedule=args.lambda_schedule,
            steps_per_mc_move=args.steps_per_iter,
            equilibration_protocol=args.equilibration_protocol,
            checkpoint_interval=args.interval,
            replica_mixing_scheme=args.replica_mixing_scheme,
        )
    else:
        raise ValueError(f"run_type {args.run_type} was not recognised")
    #upload the output file to HF
    if args.hf_repo_id is not None:
        from huggingface_hub import HfApi
        api = HfApi()
        logging.info(f"Uploading results to Hugging Face Hub at {args.hf_repo_id}")
        api.create_repo(
            repo_id=args.hf_repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=True,
        )

        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.hf_repo_id,
            repo_type="dataset",
        )
        # also upload the model_path
        api.upload_file(
            path_or_fileobj=args.model_path,
            path_in_repo=args.model_path.split("/")[-1],
            repo_id=args.hf_repo_id,
            repo_type="dataset",
        )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"MACE-MD job completed in {t2-t1:.2f} seconds")
