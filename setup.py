from setuptools import find_packages, setup
# from pybind11.setup_helpers import Pybind11Extension, build_ext
# 
package_name = 'maneuverNode'

# ext_modules = [
#     # Pybind11Extension(
#     #     "python_example",
#     #     ["maneuverNode/astar/src/main.cpp"],
#     #     # Example: passing in the version to the compiled code
#     # ),
#     Pybind11Extension(
#         "cstar",
#         ["maneuverNode/castar/src/main.cpp"],
#         # include_dirs=['/home/crarojasca/Downloads/eigen'],
#         # Example: passing in the version to the compiled code
#     ),
# ]


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
