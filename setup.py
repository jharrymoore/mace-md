"""
Various Python tools for OpenMM.
"""
import sys
from setuptools import setup, find_packages


# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []


################################################################################
# SETUP
################################################################################


setup(
    name="mace_md",
    version="0.1.0",
    author="Harry Moore",
    author_email="jhm72@cam.ac.uk",
    license="MIT",
    short_description="mace_md: Molecular dynamics frontend for running MD calculations with MACE potentials through openMM",
    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),
    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    python_requires=">=3.8",  # Python version restrictions
    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "mace-md = mace_md.cli.mace_md:main",
        ]
    },
)
