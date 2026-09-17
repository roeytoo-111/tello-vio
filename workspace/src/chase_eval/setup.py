import os

from setuptools import setup

package_name = 'chase_eval'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Roey Turgeman',
    maintainer_email='roey.turgeman@aerosentry.tech',
    description='Frozen-scenario evaluation suite, IQM/bootstrap statistics, '
                'and the replay-validation gate for chase_gym.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'evaluate = chase_eval.cli:main',
            'replay_validation = chase_eval.replay_validation:main',
        ],
    },
)
