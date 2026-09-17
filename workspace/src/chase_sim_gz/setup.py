import os
from glob import glob

from setuptools import setup

package_name = 'chase_sim_gz'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Roey Turgeman',
    maintainer_email='roey.turgeman@aerosentry.tech',
    description='Tier B: gz-sim Harmonic lockstep env, Tello stick shim, '
                'oracle detector, latency shim, sign/lag calibration.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tello_sim_shim = chase_sim_gz.tello_sim_shim:main',
            'sim_oracle_detector = chase_sim_gz.sim_oracle_detector:main',
            'latency_shim = chase_sim_gz.latency_shim:main',
            'sign_test = chase_sim_gz.sign_test:main',
            'calibrate_lag = chase_sim_gz.calibrate_lag:main',
            'gen_world = chase_sim_gz.gen_world:main',
        ],
    },
)
