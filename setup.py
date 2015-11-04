from os import path
from setuptools import setup


def read(file_name):
    return open(path.join(path.dirname(__file__), file_name)).read()


# In some cases, the numpy include path is not present by default.
# Let's try to obtain it.
try:
    from numpy import get_include
except ImportError:
    ext_include_dirs = []
else:
    ext_include_dirs = [get_include(), ]

setup_parameters = dict(
    name="semtracking",
    version='1.0',
    description="particle-tracking toolkit",
    author="Soft Matter",
    author_email="rbnvrw@gmail.com",
    url="https://bitbucket.org/softmatters/semtracking",
    install_requires=['numpy', 'scipy', 'six',
                      'pandas', 'pims', 'pims_nd2',
                      'pyyaml', 'matplotlib', 'trackpy', 'nose'],
    packages=['semtracking'],
    long_description=read('README.md')
)

setup(**setup_parameters)
