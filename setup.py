import distutils.core

distutils.core.setup(name='raise_utils',
                     version='1.1',
                     description='RAISE lab package (LGPL-3.0-or-later)',
                     author='RAISE, NC State University',
                     author_email='ryedida@ncsu.edu',
                     packages=[
                         'raise_utils.data',
                         'raise_utils.experiments',
                         'raise_utils.hyperparams',
                         'raise_utils.learners',
                         'raise_utils.metrics',
                         'raise_utils.transform',
                         'raise_utils.transform.text',
                         'raise_utils.interpret',
                         'raise_utils.hooks'
                     ]
                     )
