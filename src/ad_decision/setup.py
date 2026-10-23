from setuptools import find_packages, setup

package_name = 'ad_decision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='figguan',
    maintainer_email='contact@example.invalid',
    description='Decision making nodes for the autonomous driving project',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'behavior_state_machine = ad_decision.behavior_state_machine:main',
        ],
    },
)
