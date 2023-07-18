# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Setup script to build the TAO Toolkit package."""

import os
import setuptools

from release.python.utils import utils

PACKAGE_LIST = [
    "nvidia_tao_tf1",
    "third_party",
]

version_locals = utils.get_version_details()
setuptools_packages = []
for package_name in PACKAGE_LIST:
    setuptools_packages.extend(utils.find_packages(package_name))

setuptools.setup(
    name=version_locals['__package_name__'],
    version=version_locals['__version__'],
    description=version_locals['__description__'],
    author='NVIDIA Corporation',
    classifiers=[
        'Environment :: Console',
        # Pick your license as you wish (should match "license" above)
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: Linux',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license=version_locals['__license__'],
    keywords=version_locals['__keywords__'],
    packages=setuptools_packages,
    package_data={
        '': ['*.py', "*.pyc", "*.yaml", "*.so", "*.pdf"]
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'bpnet=nvidia_tao_tf1.cv.bpnet.entrypoint.bpnet:main',
            'classification_tf1=nvidia_tao_tf1.cv.makenet.entrypoint.makenet:main',
            'efficientdet_tf1=nvidia_tao_tf1.cv.efficientdet.entrypoint.efficientdet:main',
            'fpenet=nvidia_tao_tf1.cv.fpenet.entrypoint.fpenet:main',
            'mask_rcnn=nvidia_tao_tf1.cv.mask_rcnn.entrypoint.mask_rcnn:main',
            'multitask_classification=nvidia_tao_tf1.cv.multitask_classification.entrypoint.multitask_classification:main',
            'unet=nvidia_tao_tf1.cv.unet.entrypoint.unet:main',
            'lprnet=nvidia_tao_tf1.cv.lprnet.entrypoint.lprnet:main',
            'detectnet_v2=nvidia_tao_tf1.cv.detectnet_v2.entrypoint.detectnet_v2:main',
            'ssd=nvidia_tao_tf1.cv.ssd.entrypoint.ssd:main',
            'dssd=nvidia_tao_tf1.cv.ssd.entrypoint.ssd:main',
            'retinanet=nvidia_tao_tf1.cv.retinanet.entrypoint.retinanet:main',
            'faster_rcnn=nvidia_tao_tf1.cv.faster_rcnn.entrypoint.faster_rcnn:main',
            'yolo_v3=nvidia_tao_tf1.cv.yolo_v3.entrypoint.yolo_v3:main',
            'yolo_v4=nvidia_tao_tf1.cv.yolo_v4.entrypoint.yolo_v4:main',
            'yolo_v4_tiny=nvidia_tao_tf1.cv.yolo_v4.entrypoint.yolo_v4:main',
        ]
    }
)
