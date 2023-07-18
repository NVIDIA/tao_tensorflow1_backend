import subprocess
import os

subprocess.call(
    "{}/third_party/horovod/build_horovod.sh".format(os.getcwd()), shell=True
)
