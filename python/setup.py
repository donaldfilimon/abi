"""
ABI Framework Python Bindings

High-performance AI/ML framework with GPU acceleration and advanced algorithms.

This package provides Python bindings for the ABI (Advanced Backend Interface)
framework, enabling Python applications to leverage:

- Transformer models with GPU acceleration
- Vector databases with similarity search
- Reinforcement learning algorithms
- Federated learning coordination
- Real-time AI inference capabilities

Installation:
    pip install abi-framework

Basic usage:
    import abi

    # Create a transformer model
    model = abi.Transformer({
        'vocab_size': 30000,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6
    })

    # Vector similarity search
    db = abi.VectorDatabase(dimensions=512)
    results = db.search(query_vector, top_k=10)
"""

from setuptools import setup, Extension
import os
import sys
import subprocess


# Get the ABI framework source
def get_abi_source():
    """Get ABI framework source files for compilation"""
    abi_root = os.path.join(os.path.dirname(__file__), "abi")
    sources = []

    # Find all .zig files in the ABI framework
    for root, dirs, files in os.walk(abi_root):
        for file in files:
            if file.endswith(".zig"):
                sources.append(os.path.join(root, file))

    return sources


class ZigExtension(Extension):
    """Custom extension for Zig-based modules"""

    def __init__(self, name, sources, **kwargs):
        # Filter to only include the main binding file and dependencies
        filtered_sources = []
        for source in sources:
            if (
                "python_bindings.zig" in source
                or "mod.zig" in source
                or "main.zig" in source
            ):
                filtered_sources.append(source)

        super().__init__(name, filtered_sources, **kwargs)


def build_zig_extension():
    """Build the Zig extension"""
    try:
        # Build with Zig
        result = subprocess.run(
            [
                "zig",
                "build",
                "-Doptimize=ReleaseFast",
                "-Dtarget=x86_64-linux"
                if sys.platform.startswith("linux")
                else "-Dtarget=x86_64-macos"
                if sys.platform == "darwin"
                else "-Dtarget=x86_64-windows",
            ],
            cwd="abi",
            capture_output=True,
            text=True,
            check=True,
        )

        print("Zig build successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Zig build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


# Build the extension during setup
if not build_zig_extension():
    print("Failed to build Zig extension")
    sys.exit(1)

setup(
    name="abi-framework",
    version="0.2.0",
    author="ABI Framework Team",
    author_email="team@abi-framework.org",
    description="High-performance AI/ML framework with GPU acceleration",
    long_description=__doc__,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/abi",
    packages=["abi"],
    package_data={
        "abi": ["*.so", "*.dll", "*.dylib"],  # Compiled extensions
    },
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",  # Optional, for integration
    ],
    extras_require={
        "gpu": ["cupy>=9.0.0"],
        "dev": ["pytest>=6.0", "black", "mypy"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "abi-cli=abi.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
