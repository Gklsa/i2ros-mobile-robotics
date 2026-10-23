import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ad_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='figguan',
    maintainer_email='contact@example.invalid',
    description='Planning nodes for the autonomous driving project',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_manager = ad_planning.waypoint_manager:main',
            'trajectory_planner = ad_planning.trajectory_planner:main',
        ],
    },
)
