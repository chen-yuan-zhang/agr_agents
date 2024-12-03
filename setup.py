from setuptools import find_packages, setup

package_name = 'gr_pursuer'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools', 'gymnasium', 'multigrid'],
    zip_safe=True,
    maintainer='Cristian Rojas',
    maintainer_email='cristianlex.rojascardenas@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'gr_pursuer = gr_pursuer.gr_pursuer:main'
        ],
    }
)
