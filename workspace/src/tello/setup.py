from setuptools import setup
from glob import glob
import os

setup(
    name='tello',
    version='0.1.0',
    packages=['tello'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/tello']),
        ('share/tello', ['package.xml', 'resource/ost.txt', 'resource/ost.yaml']),
        (os.path.join('share', 'tello', 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'djitellopy',
        'PyYAML',
        'numpy',
    ],
    zip_safe=True,
    maintainer='tentone',
    maintainer_email='tentone@outlook.com',
    description='DJI Tello control package for ROS 2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tello = tello.node:main'
        ],
    },
)
