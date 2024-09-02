import os

from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = cythonize(Extension(
    name="remove_labels",
    sources=["raise_utils/transforms/remove_labels.pyx"]
))

setup(name='raise_utils',
      version='2.5.0',
      description='RAISE lab package (MIT License)',
      author='RAISE, NC State University',
      author_email='ryedida@ncsu.edu',
      long_description=open(os.path.join(
          os.path.dirname(__file__), 'README.md')).read(),
      long_description_content_type='text/markdown',
      url='https://github.com/yrahul3910/raise',
      packages=[
          'raise_utils.data',
          'raise_utils.experiments',
          'raise_utils.hyperparams',
          'raise_utils.learners',
          'raise_utils.metrics',
          'raise_utils.transforms',
          'raise_utils.transforms.text',
          'raise_utils.interpret',
          'raise_utils.hooks',
          'raise_utils.utils',
          'raise_utils'
      ],
      install_requires=[
          'scikit-learn~=1.5.1',
          'keras>=3.0.0',
          'numpy==1.26.4',
          'pandas~=2.2.2',
          'cvxopt~=1.3.1',
          'colorama~=0.4.6',
          'hyperopt~=0.2.7',
          'imblearn',
          'Cython~=0.29.24',
          'tabulate~=0.8.9',
          'statsmodels~=0.14.0'
      ],
      ext_modules=ext_modules
      )
