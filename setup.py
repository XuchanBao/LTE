#!/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path
import os

working_dir = path.abspath(path.dirname(__file__))
ROOT = os.path.abspath(os.path.dirname(__file__))

# Read the README.
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

setup(name='LTE',
      version='0.0.1',
      description='Learning to Elect',
      long_description=README,
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=['tests/*']),
      setup_requires=["numpy", "torch", "torchvision"],
      install_requires=["numpy", "scipy", "matplotlib", "test_tube", "tqdm", "munch", "imgaug", "librosa", "imageio",
                        "pytorch_lightning", "jupyter", "spaghettini", "notebook"],
      )
