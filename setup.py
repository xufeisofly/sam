from setuptools import setup, find_packages

setup(
    name='sam',
    version='0.1.0',
    entry_points={
        'console_scripts': [
            'sam = sam.main:main',
        ]
    },
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    package_dir={
        'sam': './',
    },
)