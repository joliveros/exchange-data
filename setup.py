"""
Saves cryptocurrency exchange data in realtime to influxdb
"""
from typing import List

from setuptools import find_packages, setup
import yaml


def get_deps_from_yaml(file: str='./environment.yml', exclude: List=[]):

    with open(file, 'r') as stream:
        env = yaml.load(stream)
        nested_deps = [
            dep for dep in env['dependencies'] if type(dep) == dict
        ][0]
        pip_deps = []
        if 'pip' in nested_deps:
            pip_deps = nested_deps['pip']

        pip_deps = list(set(pip_deps) - set(exclude))

        return '\n'.join(pip_deps)


setup(
    name='exchange-data',
    version='0.2.0',
    url='https://github.com/joliveros/exchange-data',
    license='No License',
    author='José Oliveros',
    author_email='jose.oliveros.1983@gmail.com',
    description='Saves cryptocurrency exchange data in realtime to influxdb',
    long_description=__doc__,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=get_deps_from_yaml(exclude=[
        'pytest',
        'pytest-cov',
        'pytest-mock',
        'pytest-vcr',
        'mock',
        'coveralls',
        'flake8'
    ]),
    tests_require=get_deps_from_yaml(),
    entry_points={
        'console_scripts': [
            'exchange-data = cli',
        ],
    },
    classifiers=[
        # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
