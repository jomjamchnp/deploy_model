"""Setup script for object_detection."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['Pillow>=1.0', 'Matplotlib>=2.1', 'Cython>=0.28.1']

setup(
    name='object_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('object_detection')],
    description='Tensorflow Object Detection Library',
)

# set PYTHONPATH=C:\Users\ACER\Desktop\deploy_model\backend;C:\Users\ACER\Desktop\deploy_model\backend\object_detection;C:\Users\ACER\Desktop\deploy_model\backend\object_detection\slim