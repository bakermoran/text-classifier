"""classifier package configuration."""

from setuptools import setup

setup(
    name='classifier',
    version='0.1.0',
    author="Baker Moran",
    author_email="bamoran99@gmail.com",
    description="A class that is a naive bayesian text classifier",
    url="https://github.com/bakermoran/text-classifier",
    packages=['classifier'],
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.16.5',
        'pandas>=0.24.2',
        'pymc3>=3.7',
        'matplotlib>=1.3.1',
        'seaborn>=0.9.0',
        'nltk'
    ],
)
