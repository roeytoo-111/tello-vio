from setuptools import setup

package_name = 'chase_gym'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'gymnasium'],
    zip_safe=True,
    maintainer='Roey Turgeman',
    maintainer_email='roey.turgeman@aerosentry.tech',
    description='Tier A chase environment and the shared train/deploy contract '
                'modules (observation, reward, latency, corruption, projection).',
    license='MIT',
)
