import os
from setuptools import setup, find_packages

BUILD_ID = os.environ.get("BUILD_BUILDID", "0")

setup(
    name="convNet",
    version="0.1" + "." + BUILD_ID,
    # Author details
    author="Jasper Ginn",
    author_email="jasperginn@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["pytorch==1.6", "numpy>=1.10"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-nunit", "pytest-cov"],
    extras_require={"develop": ["pre-commit", "bump2version"]},
)
