"""
Setup script for Hybrid LEA-Neuroplasticity Recommendation System
"""

from setuptools import setup, find_packages
import os


def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hybrid-lea-neuroplasticity-recsys",
    version="1.0.0",
    author="Saniya",
    description="A hybrid recommendation system combining LLM Environment modeling with neuroplasticity-inspired learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hybrid-lea-neuroplasticity-recsys",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
        "jupyter": [
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
        "performance": [
            "numba>=0.56.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybrid-recsys-train=scripts.train_model:main",
            "hybrid-recsys-eval=scripts.evaluate_model:main",
            "hybrid-recsys-deploy=scripts.deploy_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hybrid_recsys": [
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    
)
