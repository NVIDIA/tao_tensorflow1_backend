import argparse
import glob
import os
import subprocess
import sys

from ci.utils import CI
from nvidia_tao_tf1.core.utils.path_utils import expand_path

def execute_command(command_args):
    """Execute the shell command."""
    for command in command_args:
        try:
            subprocess.call(
                command,
                shell=True,
                stdout=sys.stdout
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Command {command} failed with error {exc}") from exc


def parse_command_line(cl_args=None):
    """Parse builder command line"""
    parser = argparse.ArgumentParser(
        prog="build_kernel",
        description="Build TAO custom ops."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force build the kernel."
    )
    parser.add_argument(
        "--op_names",
        type=str,
        nargs="+",
        help="The name of the op to build.",
        default="all"
    )
    return vars(parser.parse_args(cl_args))


def list_op_libraries(path, op_names):
    """List of library paths."""
    if os.path.isdir(expand_path(path)):
        op_list = [
            item for item in os.listdir(expand_path(path)) if
            os.path.isdir(
                os.path.join(path, item)
            )
        ]
    if op_names == "all":
        return op_list
    return [item for item in op_list if item in op_names]


def build_ops(op_list, path, force=False):
    """Build custom ops."""
    prev_dir = os.getcwd()
    for op in op_list:
        print(f"Building op {op}")
        build_command = []
        if force:
            build_command.append("make clean")
        build_command.append("make")
        op_path = os.path.join(path, op)
        if os.path.isdir(op_path):
            os.chdir(op_path)
        if not os.path.exists(os.path.join(op_path, "Makefile")):
            continue
        execute_command(build_command)
        assert os.path.isfile(os.path.join(op_path, f"../op_{op}.so")), (
            f"\'{op}\' build failed."
        )
    os.chdir(prev_dir)


def main(cl_args=sys.argv[1:]):
    """Run kernel builder."""
    args = parse_command_line(cl_args=cl_args)
    force_build = args["force"]
    op_names = args["op_names"]
    env_var = "NV_TAO_TF_TOP"
    if CI:
        env_var = "CI_PROJECT_DIR"
    if "WORKSPACE" not in os.environ.keys():
        os.environ["WORKSPACE"] = os.getenv(env_var, "/workspace/tao-tf1")
    h_path = os.getenv(env_var, "/workspace/tao-tf1")
    glob_string = os.path.abspath(os.path.expanduser(f"{h_path}/nvidia_tao_tf1/**/*/lib"))
    kernel_lib_root = glob.glob(
        glob_string,
        recursive=True
    )
    for lib_path in kernel_lib_root:
        op_list = list_op_libraries(lib_path, op_names)
        build_ops(op_list, lib_path, force=force_build)


if __name__ == "__main__":
    main()