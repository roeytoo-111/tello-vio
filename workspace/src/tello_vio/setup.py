import os
from glob import glob

from setuptools import setup

package_name = 'tello_vio'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, package_name + '.nodes'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML'],
    zip_safe=True,
    maintainer='Roey Turgeman',
    maintainer_email='roey.turgeman@aerosentry.tech',
    description='Visual-inertial odometry for the DJI Tello.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vio = tello_vio.nodes.vio_node:main',
            'imu_calib = tello_vio.nodes.imu_calib_node:main',
            'camera_imu_calib = tello_vio.nodes.camera_imu_calib_node:main',
            'map_align = tello_vio.nodes.map_align_node:main',
        ],
    },
)
