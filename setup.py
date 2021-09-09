from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="SPECTRA",
    version="0.0.1",
    description="SPECTRA",
    long_description=readme,
    author="Nuno M Guerreiro",
    author_email="miguelguerreironuno@gmail.com",
    url="https://github.com/deep-spin/spectra-rationalization",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
    data_files=["LICENSE"],
    zip_safe=False,
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
