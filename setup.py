import io
import os
from pathlib import Path
from typing import List
from setuptools import find_packages, setup


# Metadata of package
NAME = 'Prediction_Model'
DESCRIPTION = 'Cemenet Strenght Prediction Model'
URL = 'https://github.com/anonhan'
EMAIL = 'sahiltheanalyst@gmail.com'
AUTHOR = 'Sahil Sharma'
REQUIRES_PYTHON = '>=3.10.0'

pwd = os.path.abspath(os.path.dirname(__file__))

# Get the list of packages to be installed
def get_requirements()->List[str]:
    """
    This function will return list of requirements
    """
    requirement_list:List[str] = []
    return requirement_list

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'src' /NAME
about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'Prediction_Model': ['VERSION']},
    install_requires=get_requirements(),
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    entry_points={
    'console_scripts': [
        'main=Prediction_Model.main:main',
    ]}
)