from distutils.core import setup

setup(name='raise_utils',
      version='1.0',
      description='RAISE lab package',
      author='RAISE, NC State University',
      author_email='ryedida@ncsu.edu',
      packages=[
          'raise.data',
          'raise.experiments',
          'raise.hyperparams',
          'raise.learners',
          'raise.metrics',
          'raise.transform'
      ]
      )
