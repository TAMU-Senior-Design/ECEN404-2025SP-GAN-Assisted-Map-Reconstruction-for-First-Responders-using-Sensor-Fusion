from setuptools import setup

package_name = 'pointcloud_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Quinn',
    maintainer_email='your@email.com',
    description='Saves point cloud from topic to .obj file',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'save_pointcloud = pointcloud_tools.save_pointcloud:main'
        ],
    },
)
