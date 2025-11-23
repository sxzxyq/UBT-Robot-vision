from setuptools import setup, find_packages

package_name = 'image_pipeline_ros'

setup(
    name=package_name,
    version='0.0.1',
    # 告诉 setuptools，所有包的根目录都在上一级 (即 'src' 目录)
    package_dir={'': '..'},
    # 精确地告诉 find_packages 要包含哪些包
    # 它会在 'src' 目录下寻找这些名字的文件夹
    packages=[
        'image_pipeline_ros', 
        'ImagePipeline',
        'cutie',
        'sam2',
        # 你可能还需要包含这些库的子包，使用 find_packages 更安全
        *find_packages(where='..', include=['ImagePipeline*', 'cutie*', 'sam2*'])
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@nvidia.com',
    description='A self-contained ROS 2 vision pipeline.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pipeline_node = image_pipeline_ros.image_pipeline_node:main',
        ],
    },
)