# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Utilties to send data to the TAO Toolkit Telemetry Remote Service."""

import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib

import requests
import urllib3

from nvidia_tao_tf1.cv.common.telemetry.nvml_utils import get_device_details

logger = logging.getLogger(__name__)

TELEMETRY_TIMEOUT = int(os.getenv("TELEMETRY_TIMEOUT", "30"))


def get_url_from_variable(variable, default=None):
    """Get the Telemetry Server URL."""
    url = os.getenv(variable, default)
    return url


def url_exists(url):
    """Check if a URL exists.

    Args:
        url (str): String to be verified as a URL.

    Returns:
        valid (bool): True/Falso
    """
    url_request = urllib.request.Request(url)
    url_request.get_method = lambda: 'HEAD'
    try:
        urllib.request.urlopen(url_request)
        return True
    except urllib.request.URLError:
        return False


def get_certificates():
    """Download the cacert.pem file and return the path.

    Returns:
        path (str): UNIX path to the certificates.
    """
    certificates_url = get_url_from_variable("TAO_CERTIFICATES_URL")
    if not url_exists(certificates_url):
        raise urllib.request.URLError("Url for the certificates not found.")
    tmp_dir = tempfile.mkdtemp()
    download_command = "wget {} -P {} --quiet".format(
        certificates_url,
        tmp_dir
    )
    try:
        subprocess.check_call(
            download_command, shell=True, stdout=sys.stdout
        )
    except subprocess.CalledProcessError:
        raise urllib.request.URLError("Download certificates.tar.gz failed.")
    tarfile_path = os.path.join(tmp_dir, "certificates.tar.gz")
    assert tarfile.is_tarfile(tarfile_path), (
        "The downloaded file isn't a tar file."
    )
    with tarfile.open(name=tarfile_path, mode="r:gz") as tar_file:
        filenames = tar_file.getnames()
        for memfile in filenames:
            member = tar_file.getmember(memfile)
            tar_file.extract(member, tmp_dir)
    file_list = [item for item in os.listdir(tmp_dir) if item.endswith(".pem")]
    assert file_list, (
        f"Didn't get pem files. Directory contents {file_list}"
    )
    return tmp_dir


def send_telemetry_data(network, action, gpu_data, num_gpus=1, time_lapsed=None, pass_status=False):
    """Wrapper to send TAO telemetry data.

    Args:
        network (str): Name of the network being run.
        action (str): Subtask of the network called.
        gpu_data (dict): Dictionary containing data about the GPU's in the machine.
        num_gpus (int): Number of GPUs used in the job.
        time_lapsed (int): Time lapsed.
        pass_status (bool): Job passed or failed.

    Returns:
        No explicit returns.
    """
    urllib3.disable_warnings(urllib3.exceptions.SubjectAltNameWarning)
    if os.getenv('TELEMETRY_OPT_OUT', "no").lower() in ["no", "false", "0"]:
        url = get_url_from_variable("TAO_TELEMETRY_SERVER")
        data = {
            "version": os.getenv("TAO_TOOLKIT_VERSION", "4.0.0"),
            "action": action,
            "network": network,
            "gpu": [device["name"] for device in gpu_data[:num_gpus]],
            "success": pass_status
        }
        if time_lapsed is not None:
            data["time_lapsed"] = time_lapsed
        certificate_dir = get_certificates()
        cert = ('client-cert.pem', 'client-key.pem')
        requests.post(
            url,
            json=data,
            cert=tuple([os.path.join(certificate_dir, item) for item in cert]),
            timeout=TELEMETRY_TIMEOUT
        )
        logger.debug("Telemetry data posted: \n{}".format(
            json.dumps(data, indent=4)
        ))
        shutil.rmtree(certificate_dir)

if __name__ == "__main__":
    try:
        print("Testing telemetry data ping.")
        gpu_data = []
        for device in get_device_details():
            gpu_data.append(device.get_config())
            print(device)
        send_telemetry_data(
            "detectnet_v2",
            "train",
            gpu_data,
            1
        )
    except Exception as e:
        logger.warning(
            "Telemetry data failed with error:\n{}".format(e)
        )
