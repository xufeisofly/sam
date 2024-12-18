from setuptools import setup, find_packages

setup(
    name='sam',
    version='0.1.0',
    entry_points={
        'console_scripts': [
            'sam = sam.main:main',
            'show-box = sam.show_box:show_box',
            'list-gpu-ids = sam.main:list_gpu_ids',
            'fix-detection-file = sam.fix_detection_file:main',
            'check-detection-file = sam.fix_detection_file:check_detection_file'
        ]
    },
    packages=find_packages(),
    install_requires=[
    ],
)