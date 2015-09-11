import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# In some cases, the numpy include path is not present by default.
# Let's try to obtain it.
try:
    import numpy
except ImportError:
    ext_include_dirs = []
else:
    ext_include_dirs = [numpy.get_include(), ]

setup_parameters = dict(
    name="semtracking",
    version='1.0',
    description="particle-tracking toolkit",
    author="Soft Matter",
    author_email="rbnvrw@gmail.com",
    url="https://bitbucket.org/softmatters/semtracking",
    install_requires=['numpy>=1.7', 'scipy>=0.12', 'six>=1.8',
                      'pandas>=0.12', 'pims',
                      'pyyaml', 'matplotlib', 'trackpy'],
    packages=['semtracking'],
    long_description=read('README.md'),
)

setup(**setup_parameters, requires=['scipy', 'numpy', 'pandas'])
