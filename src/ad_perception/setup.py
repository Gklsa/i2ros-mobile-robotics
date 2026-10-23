from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ad_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='figguan',
    maintainer_email='contact@example.invalid',
    description='Camera-based perception for the autonomous driving project',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sensor_tf_broadcaster = ad_perception.sensor_tf_broadcaster:main',
            'traffic_light_detector = ad_perception.traffic_light_detector:main',
            'obstacle_detector = ad_perception.obstacle_detector:main',
            'occupancy_grid_generator = '
            'ad_perception.occupancy_grid_generator:main',
            # Historical dataset-capture tool. Not installed as a formal
            # runtime executable because the models are already trained.
            # 'trafficlight_dataset_collector = '
            # 'ad_perception.trafficlight_dataset_collector:main',
            'yolo_car_detector = ad_perception.yolo_car_detector:main',
        ],
    },
)
