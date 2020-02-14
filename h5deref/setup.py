import setuptools
import h5deref

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [i.strip() for i in open("requirements.txt").readlines()]

setuptools.setup(
    name="h5deref",
    version=h5deref.__version__,
    author="Sören J Zapp",
    author_email="dev.szapp@gmail.com",
    description="Load HDF5 files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/szapp/h5deref",
    license='MIT',
    requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6',
)
