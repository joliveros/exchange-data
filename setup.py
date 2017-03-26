"""
Saves bitmex data in realtime to influxdb
"""
from setuptools import find_packages, setup

dependencies = ['click',
                'bitmex-websocket',
                'influxdb',
                'future',
                'asyncio']

setup(
    name='save-bitmex-data',
    version='0.1.0',
    url='https://github.com/joliveros/save-bitmex-data',
    license='BSD',
    author='José Oliveros',
    author_email='jose.oliveros.1983@gmail.com',
    description='Saves bitmex data in realtime to influxdb',
    long_description=__doc__,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'save-bitmex-data = save_bitmex_data.cli:main',
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
