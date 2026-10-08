from glob import glob

from setuptools import setup

package_name = 'depth_processing_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.*')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='masonbp',
    maintainer_email='mbpeterson70@gmail.com',
    description='Online correction of the ZED stereo depth bias, calibrated from the ground plane.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_scale_correction_node = depth_processing_ros.depth_scale_correction_node:main',
        ],
    },
)
