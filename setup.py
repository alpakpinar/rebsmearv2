from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = 'rebsmearv2',
    version = '0.0.1',
    url = 'https://github.com/bu-cms/bucoffea',
    author = 'Alp Akpinar',
    author_email = 'aakpinar@cern.ch',
    description = 'Implementation of rebalance & smear',
    packages = find_packages(),    
    install_requires = requirements,
    scripts=[
        "rebsmearv2/execute/rebexec"
    ]
)