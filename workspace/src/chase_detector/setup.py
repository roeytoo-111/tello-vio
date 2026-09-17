import os
from glob import glob

from setuptools import setup

package_name = 'chase_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.nodes'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Roey Turgeman',
    maintainer_email='roey.turgeman@aerosentry.tech',
    description='YOLO drone detection on the Tello video stream: inference '
                'node, live viewer, and dataset capture.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detector = chase_detector.nodes.detector_node:main',
            'viewer = chase_detector.nodes.viewer_node:main',
            'capture = chase_detector.nodes.capture_node:main',
        ],
    },
)
