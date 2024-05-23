from setuptools import setup, Extension
from Cython.Build import cythonize
import os

ext_modules = cythonize(Extension(
    name="remove_labels",
    sources=["raise_utils/transforms/remove_labels.pyx"]
))

setup(name='raise_utils',
      version='2.3.0',
      description='RAISE lab package (LGPL-3.0-or-later)',
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
          'scikit-learn>=0.23.2',
          'keras>=3.0.0',
          'numpy>=1.19.2',
          'pandas',
          'cvxopt',
          'colorama',
          'hyperopt',
          'imblearn',
          'Cython',
          'tabulate',
          'statsmodels'
      ],
      ext_modules=ext_modules
      )
