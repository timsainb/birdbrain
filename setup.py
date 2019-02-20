from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='birdbrain',
    packages=find_packages(),
    version='0.1.1',
    description='Tools for using the songbird brain atlas',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/timsainb/birdbrain",
    author='Tim Sainburg',
    license='BSD-3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        'Framework :: IPython',
        'Intended Audience :: Scientists',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics',
    ],
    install_requires=[
        'ipywidgets>=7.0.1',
        'ipydatawidgets',
        'traittypes',
        'traitlets',
        'numpy>=1.11.0',
        'k3d',
        'nibabel',
        'nipy',
        'scipy',
        'matplotlib',
        'vtk'
    ],

)
