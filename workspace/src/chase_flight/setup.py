import os
from glob import glob

from setuptools import setup

package_name = 'chase_flight'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.nodes'],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Roey Turgeman',
    maintainer_email='roey.turgeman@aerosentry.tech',
    description='Real-time flight stack for the vision-based drone chase.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'chase_state = chase_flight.nodes.state_node:main',
            'chase_policy = chase_flight.nodes.policy_node:main',
            'chase_depth_rule = chase_flight.nodes.depth_rule_node:main',
            'chase_behaviour = chase_flight.nodes.behaviour_node:main',
            'chase_mixer = chase_flight.nodes.mixer_node:main',
            'chase_safety = chase_flight.nodes.safety_node:main',
        ],
    },
)
