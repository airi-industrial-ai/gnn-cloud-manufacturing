from setuptools import find_packages, setup

setup(
    name='cloudmanufacturing',
    packages=find_packages(),
    version='0.0.1',
    description='GNN for cloud manufacturing',
    url='https://github.com/airi-industrial-ai/gnn-claud-manufacturing',
    author='AIRI, Industrial AI',
    install_requires=[
        'pandas',
        'tqdm',
        'pytest',
        'dgl',
        'torch',
        'numpy',
        'mip',
        'scikit-learn',
        'openpyxl',
    ],
)