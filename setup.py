# install using 'pip install -e .'

from setuptools import setup, find_packages

setup(
    name='pointnet',
    version='0.0.1',
    packages=find_packages(where='models'),
    package_dir={'': 'models'},
    install_requires=[
        'torch',
        'tqdm',
        'plyfile'
    ],
)