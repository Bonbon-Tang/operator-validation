from setuptools import setup, find_packages

setup(
    name="operator-validation",
    version="0.1.0",
    description="Multi-backend operator validation framework for AI accelerators",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "prettytable>=3.8.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "mlu": ["torch-mlu>=1.0.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "operator-validate=operator_validation.main:main",
        ],
    },
    author="Your Name",
    author_email="you@example.com",
    license="MIT",
)
