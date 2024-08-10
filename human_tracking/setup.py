from setuptools import setup
import os
from glob import glob

package_name = 'human_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'description'), glob(os.path.join('description', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chetan',
    maintainer_email='chetan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'rgbd = human_tracking.rgbd_subscription:main',
        	'state = human_tracking.state_estimation:main'
        ],
    },
)
