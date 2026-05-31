import os

from setuptools import Extension, setup

ext_modules = [Extension(
    name="raise_utils.transforms.remove_labels",
    sources=["raise_utils/transforms/remove_labels.pyx"]
)]

setup(author='Rahul Yedida',
      author_email='ryedida@ncsu.edu',
      long_description=open(os.path.join(
          os.path.dirname(__file__), 'README.md')).read(),
      long_description_content_type='text/markdown',
      url='https://github.com/yrahul3910/raise',
      ext_modules=ext_modules
      )
