from setuptools import setup, find_packages

setup(
    name='sam',
    version='0.1.0',
    # entry_points={
    #     'console_scripts': [
    #         'super-macro-server = server:main',
    #         'transfer-file = server:bin_to_csv',
    #         'transfer-motor-file = server:transfer_motor_file',
    #         'super-macro-client = client:run',
    #         'shell-client = client.shell_client:run_shell_client',
    #     ]
    # },
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    # package_dir={
    #     'server': './server',
    #     'client': './client',
    # },
)