# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export a classification model."""

# import build_command_line_parser as this is needed by entrypoint
from nvidia_tao_tf1.cv.common.export.app import build_command_line_parser as global_parser # noqa pylint: disable=W0611
from nvidia_tao_tf1.cv.common.export.app import run_export
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.makenet.export.classification_exporter import ClassificationExporter \
    as Exporter


def build_command_line_parser(parser=None):
    """Simple function to build the command line parser."""
    args_parser = global_parser(parser=parser)
    args_parser.add_argument(
        "--classmap_json",
        help="UNIX path to classmap.json file generated during classification <train>",
        default=None,
        type=str,
    )
    return args_parser


def parse_command_line(args=None):
    """Parse command line arguments."""
    parser = build_command_line_parser(parser=None)
    return vars(parser.parse_known_args(args)[0])


def main(args=None):
    """Run export for classification."""
    try:
        args = parse_command_line(args=args)

        # Forcing export to ONNX by default.
        backend = 'onnx'
        run_export(Exporter, args=args, backend=backend)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Export finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Export was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == "__main__":
    main()
