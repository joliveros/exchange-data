"""
Saves cryptocurrency exchange data in realtime to influxdb
"""
from setuptools import find_packages, setup
from pip.req import parse_requirements
from pip.download import PipSession
from os.path import realpath


def get_reqs_from_file(file):
    file_path = realpath(file)

    # parse_requirements() returns generator of pip.req.InstallRequirement objects
    install_reqs = parse_requirements(file_path, session=PipSession)

    # reqs is a list of requirement
    # e.g. ['django==1.5.1', 'mezzanine==1.4.6']
    return [str(ir.req) for ir in install_reqs]


setup(
    name='exchange-data',
    version='0.2.0',
    url='https://github.com/joliveros/exchange-data',
    license='BSD',
    author='José Oliveros',
    author_email='jose.oliveros.1983@gmail.com',
    description='Saves cryptocurrency exchange data in realtime to influxdb',
    long_description=__doc__,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=get_reqs_from_file('./requirements.txt'),
    tests_require=get_reqs_from_file('./requirements-test.txt'),
    entry_points={
        'console_scripts': [
            'exchange-data = exchange_data.cli:main',
        ],
    },
    classifiers=[
        # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        'Development Status :: 4 - Beta',
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
