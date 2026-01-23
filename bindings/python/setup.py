"""
ABI Framework Python Bindings - Setup Script

Installation:
    pip install .

Development installation:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
import os

# Read the README if it exists
long_description = ""
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        # Skip YAML front matter
        content = f.read()
        if content.startswith("---"):
            # Find end of front matter
            end = content.find("---", 3)
            if end != -1:
                content = content[end + 3:].strip()
        long_description = content

setup(
    name="abi",
    version="0.4.0",
    author="ABI Framework Team",
    author_email="team@abi-framework.dev",
    description="Python bindings for the ABI high-performance AI and vector database framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/donaldfilimon/abi",
    project_urls={
        "Documentation": "https://github.com/donaldfilimon/abi#readme",
        "Bug Tracker": "https://github.com/donaldfilimon/abi/issues",
        "Source Code": "https://github.com/donaldfilimon/abi",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
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
            "isort>=5.0",
        ],
        "numpy": [
            "numpy>=1.20",
        ],
        "all": [
            "numpy>=1.20",
        ],
    },
    package_data={
        "abi": ["*.so", "*.dylib", "*.dll", "py.typed"],
    },
    entry_points={
        "console_scripts": [
            "abi-py=abi.cli:main",
        ],
    },
    keywords=[
        "ai",
        "llm",
        "vector-database",
        "embeddings",
        "gpu",
        "cuda",
        "vulkan",
        "machine-learning",
        "neural-network",
        "inference",
    ],
)
