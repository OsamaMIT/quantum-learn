from setuptools import setup, find_packages
import os

# Ensure README exists to avoid FileNotFoundError
long_description = open("README.md").read() if os.path.exists("README.md") else ""

setup(
    name="quantum-learn",
    version="0.0.1",
    description="quantum-learn: quantum machine learning in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OsamaMIT",
    author_email="author@example.com", # Placeholder
    url="https://github.com/OsamaMIT/quantum-learn",  # Repository URL
    packages=find_packages(),            # Automatically find packages in the directory
    install_requires=[
        "pennylane",
        "pandas",
        "matplotlib",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",  # Optional: Indicates early-stage development
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.6',
    license="MIT"
)
