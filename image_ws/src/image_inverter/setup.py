from setuptools import find_packages, setup

package_name = 'image_inverter'

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
    maintainer='nvidia',
    maintainer_email='kai.yang@x-humanoid.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'helmet_tracker = image_inverter.helmet_tracker_node:main',
            'depth_to_pointcloud = image_inverter.depth_to_pointcloud_node:main',
            'fusion = image_inverter.fusion_node:main',
            'helmet_transformer = image_inverter.helmet_tf_transformer:main',
            'kalman_filter_node = image_inverter.kalman_filter_node:main',
            'secondary_verifier = image_inverter.secondary_verifier_node:main',
            'object_counter = image_inverter.object_counter_node:main',
        ],
    },
)
