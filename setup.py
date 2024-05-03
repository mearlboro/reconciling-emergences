from setuptools import find_packages, setup
setup(
    name="reconciling-emergences",
    version="0.0.2",
    packages=find_packages(include=["emergence", "emergence.*"]),
)
