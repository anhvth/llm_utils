# setup.py
from setuptools import setup, find_packages

setup(
    name="llm_data_utils",
    version="0.1.0",
    author="AnhVTH",
    author_email="anhvth.226@gmail.com",
    description="A small package for data processing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas'
    ],
)
