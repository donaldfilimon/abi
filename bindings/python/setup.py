"""
ABI Framework Python Bindings - Setup Script

Installation:
    pip install .

Development installation:
    pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read the README if it exists
long_description = ""
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="abi",
    version="0.3.0",
    author="ABI Framework Team",
    description="Python bindings for the ABI high-performance AI and vector database framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/donaldfilimon/abi",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "mypy>=1.0",
            "ruff>=0.1.0",
        ],
        "numpy": [
            "numpy>=1.20",
        ],
    },
    package_data={
        "abi": ["*.so", "*.dylib", "*.dll"],
    },
    entry_points={
        "console_scripts": [
            "abi-py=abi.cli:main",
        ],
    },
)
