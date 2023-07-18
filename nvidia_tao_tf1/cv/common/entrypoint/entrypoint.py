# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""TLT command line wrapper to invoke CLI scripts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import logging
import os
import pkgutil
import shlex
import subprocess
import sys
from time import time

import nvidia_tao_tf1.cv.common.no_warning  # noqa pylint: disable=W0611
from nvidia_tao_tf1.cv.common.telemetry.nvml_utils import get_device_details
from nvidia_tao_tf1.cv.common.telemetry.telemetry import send_telemetry_data

MULTIGPU_SUPPORTED_TASKS = ["train"]
RELEASE = True

logger = logging.getLogger(__name__)


def get_modules(package):
    """Function to get module supported tasks.

    This function lists out the modules in the nvidia_tao_tf1.cv.X.scripts package
    where the module subtasks are listed, and walks through it to generate a dictionary
    of tasks, parser_function and path to the executable.

    Args:
        No explicit args.

    Returns:
        modules (dict): Dictionary of modules.
    """
    modules = {}
    module_path = package.__path__
    tasks = [item[1] for item in pkgutil.walk_packages(module_path)]
    for task in sorted(tasks, key=str.lower, reverse=True):
        module_name = package.__name__ + '.' + task
        module = importlib.import_module(module_name)
        module_details = {
            "module_name": module_name,
            "build_parser": getattr(
                module,
                "build_command_line_parser") if hasattr(
                    module,
                    "build_command_line_parser"
                ) else None,
            "runner_path": os.path.abspath(
                module.__file__
            )
        }
        modules[task] = module_details
    return modules


def build_command_line_parser(package_name, modules=None):
    """Simple function to build command line parsers.

    This function scans the dictionary of modules determined by the
    get_modules routine and builds a chained parser.

    Args:
        modules (dict): Dictionary of modules as returned by the get_modules function.

    Returns:
        parser (argparse.ArgumentParser): An ArgumentParser class with all the
            subparser instantiated for chained parsing.
    """
    parser = argparse.ArgumentParser(
        package_name,
        add_help=True,
        description="Transfer Learning Toolkit"
    )
    parser.add_argument(
        "--num_processes",
        "-np",
        type=int,
        default=-1,
        help=("The number of horovod child processes to be spawned. "
              "Default is -1(equal to --gpus)."),
        required=False
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help="The number of GPUs to be used for the job.",
        required=False,
    )
    parser.add_argument(
        '--gpu_index',
        type=int,
        nargs="+",
        help="The indices of the GPU's to be used.",
        default=None
        )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Flag to enable Auto Mixed Precision."
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help="Path to the output log file.",
        required=False,
    )
    parser.add_argument(
        "--mpirun-arg",
        type=str,
        default="-x NCCL_IB_HCA=mlx5_4,mlx5_6,mlx5_8,mlx5_10 -x NCCL_SOCKET_IFNAME=^lo,docker",
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--multi-node',
        action='store_true',
        default=False,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--launch_cuda_blocking",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS
    )

    # module subparser for the respective tasks.
    module_subparsers = parser.add_subparsers(title="tasks")
    for task, details in modules.items():
        if not details['build_parser']:
            logger.debug("Parser for task {} wasn't built.".format(
                task
            ))
            continue
        subparser = module_subparsers.add_parser(
            task,
            parents=[parser],
            add_help=False)
        subparser = details['build_parser'](subparser)
    return parser


def format_command_line_args(args):
    """Format command line args from command line.

    Args:
        args (dict): Dictionary of parsed command line arguments.

    Returns:
        formatted_string (str): Formatted command line string.
    """
    assert isinstance(args, dict), (
        "The command line args should be formatted to a dictionary."
    )
    formatted_string = ""
    for arg, value in args.items():
        if arg in ["gpus", "gpu_index", "log_file", "use_amp",
                   "multi_node", "mpirun_arg", "num_processes",
                   "launch_cuda_blocking"]:
            continue
        # Fix arguments that defaults to None, so that they will
        # not be converted to string "None". Simply drop args
        # that have value None.
        # For example, export output_file arg and engine_file arg
        # same for "" for cal_image_dir in export.
        if value in [None, ""]:
            continue
        if isinstance(value, bool):
            if value:
                formatted_string += "--{} ".format(arg)
        elif isinstance(value, list):
            formatted_string += "--{} {} ".format(
                arg, ' '.join(value)
            )
        else:
            formatted_string += "--{} {} ".format(
                arg, value
            )
    return formatted_string


def check_valid_gpus(num_gpus, gpu_ids):
    """Check if the number of GPU's called and IDs are valid.

    This function scans the machine using the nvidia-smi routine to find the
    number of GPU's and matches the id's and num_gpu's accordingly.

    Once validated, it finally also sets the CUDA_VISIBLE_DEVICES env variable.

    Args:
        num_gpus (int): Number of GPUs alloted by the user for the job.
        gpu_ids (list(int)): List of GPU indices used by the user.

    Returns:
        No explicit returns
    """
    # Ensure the gpu_ids are all different, and sorted
    gpu_ids = sorted(list(set(gpu_ids)))
    assert num_gpus > 0, "At least 1 GPU required to run any task."
    num_gpus_available = str(subprocess.check_output(["nvidia-smi", "-L"])).count("UUID")
    max_id = max(gpu_ids)
    assert min(gpu_ids) >= 0, (
        "GPU ids cannot be negative."
    )
    assert len(gpu_ids) == num_gpus, (
        "The number of GPUs ({}) must be the same as the number of GPU indices"
        " ({}) provided.".format(
            gpu_ids,
            num_gpus
        )
    )
    assert max_id < num_gpus_available and num_gpus <= num_gpus_available, (
        "Checking for valid GPU ids and num_gpus."
    )
    cuda_visible_devices = ",".join([str(idx) for idx in gpu_ids])
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices


def get_env_variables(use_amp):
    """Simple function to get env variables for the run command."""
    env_variable = ""

    amp_enable = "TF_ENABLE_AUTO_MIXED_PRECISION=0"
    if use_amp:
        amp_enable = "TF_ENABLE_AUTO_MIXED_PRECISION=1"
    env_variable += amp_enable
    return env_variable


def set_gpu_info_single_node(num_gpus, gpu_ids):
    """Set gpu environment variable for single node."""
    check_valid_gpus(num_gpus, gpu_ids)

    env_variable = ""
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is not None:
        env_variable = " CUDA_VISIBLE_DEVICES={}".format(
            visible_devices
        )

    return env_variable


def launch_job(package, package_name, cl_args=None):
    """Wrap CLI builders.

    This function should be included inside package entrypoint/*.py

    import sys
    import nvidia_tao_tf1.cv.X.scripts
    from nvidia_tao_tf1.cv.common.entrypoint import launch_job

    if __name__ == "__main__":
        launch_job(nvidia_tao_tf1.cv.X.scripts, "X", sys.argv[1:])
    """
    # Configure the logger.
    verbosity = "INFO"
    if not RELEASE:
        verbosity = "DEBUG"
    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
        level=verbosity
    )

    # build modules
    modules = get_modules(package)
    parser = build_command_line_parser(package_name, modules)

    # parse command line arguments to module entrypoint script.
    args = vars(parser.parse_args(cl_args))
    num_gpus = args["gpus"]
    assert num_gpus > 0, "At least 1 GPU required to run any task."
    np = args["num_processes"]
    # np defaults to num_gpus if < 0
    if np < 0:
        np = num_gpus
    gpu_ids = args["gpu_index"]
    use_amp = args['use_amp']
    multi_node = args['multi_node']
    mpirun_arg = args['mpirun_arg']
    launch_cuda_blocking = args["launch_cuda_blocking"]
    process_passed = True
    if gpu_ids is None:
        gpu_ids = range(num_gpus)

    log_file = sys.stdout
    if args['log_file'] is not None:
        log_file = os.path.realpath(args['log_file'])
        log_root = os.path.dirname(log_file)
        if not os.path.exists(log_root):
            os.makedirs(log_root)

    # Get the task to be called from the raw command line arguments.
    task = None
    for arg in sys.argv[1:]:
        if arg in list(modules.keys()):
            task = arg
        break
    # Either data parallelism or model parallelism, multi-gpu should only
    # apply to training task
    if num_gpus > 1:
        assert task in MULTIGPU_SUPPORTED_TASKS, (
            "Please use only 1 GPU for the task {}. Only the following tasks "
            "are supported to run with multiple GPUs, {}".format(
                task,
                MULTIGPU_SUPPORTED_TASKS)
        )
    # Check for validity in terms of GPU handling and available resources.
    mpi_command = ""
    if np > 1:
        assert num_gpus > 1, (
            "Number of GPUs must be > 1 for data parallelized training(np > 1)."
        )
        mpi_command = f'mpirun -np {np} --oversubscribe --bind-to none --allow-run-as-root -mca pml ob1 -mca btl ^openib'
        if multi_node:
            mpi_command += " " + mpirun_arg
    if use_amp:
        assert task == "train", (
            "AMP is currently supported only for training."
        )

    # Format final command.
    env_variables = get_env_variables(use_amp)
    if not multi_node:
        env_variables += set_gpu_info_single_node(num_gpus, gpu_ids)
    formatted_args = format_command_line_args(args)
    task_command = "python {}".format(modules[task]["runner_path"])
    if launch_cuda_blocking:
        task_command = f"CUDA_LAUNCH_BLOCKING=1 {task_command}"

    run_command = "{} bash -c '{} {} {}'".format(
        mpi_command,
        env_variables,
        task_command,
        formatted_args)

    logger.debug("Run command: {}".format(run_command))

    start_mark = time()
    try:
        if isinstance(log_file, str):
            with open(log_file, "a") as lf:
                subprocess.run(
                    shlex.split(run_command),
                    shell=False,
                    env=os.environ,
                    stdout=lf,
                    stderr=lf,
                    check=True
                )
        else:
            subprocess.run(
                shlex.split(run_command),
                shell=False,
                env=os.environ,
                stdout=log_file,
                stderr=log_file,
                check=True
            )
    except (KeyboardInterrupt, SystemExit):
        print("Command was interrupted.")
        process_passed = True
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(f"TAO Toolkit task: {task} failed with error:\n{e.output}")
        process_passed = False
    end_mark = time()
    time_lapsed = int(end_mark - start_mark)

    try:
        gpu_data = []
        logger.debug("Gathering GPU data for TAO Toolkit Telemetry.")
        for device in get_device_details():
            gpu_data.append(device.get_config())
        logger.debug("Sending data to the TAO Telemetry server.")
        send_telemetry_data(
            package_name,
            task,
            gpu_data,
            num_gpus=num_gpus,
            time_lapsed=time_lapsed,
            pass_status=process_passed
        )
    except Exception as e:
        print("Telemetry data couldn't be sent, but the command ran successfully.")
        print(f"[WARNING]: {e}")
        pass

    if not process_passed:
        print("Execution status: FAIL")
        sys.exit(-1)  # returning non zero return code from the process.

    print("Execution status: PASS")
