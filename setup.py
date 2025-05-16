from setuptools import setup, find_packages
import os

setup(
    name='llms',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'flax',
        'datasets',
        'optax',
        'tensorflow',
        'tensorflow-datasets',
        'transformers',
        'wandb',
    ],
    author='David Ulloa, Gabriel Lucchesi',
    author_email='dulloa6310@gmail.com, gabrielhubnerlucchesi@gmail.com',
    description='LLMs implemented from scratch, using Jax / Flax.',
    long_description=open('README.md').read(
    ) if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)
