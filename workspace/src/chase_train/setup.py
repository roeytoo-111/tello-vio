import os
from glob import glob

from setuptools import setup

package_name = 'chase_train'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'gymnasium', 'torch'],
    zip_safe=True,
    maintainer='Roey Turgeman',
    maintainer_email='roey.turgeman@aerosentry.tech',
    description='The one TD3/DDPG trainer with the three-arm flags, the '
                'checkpoint contract, and the run matrix.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'train = chase_train.train:main',
            'run_matrix = chase_train.run_matrix:main',
        ],
    },
)
