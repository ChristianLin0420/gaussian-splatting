from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gaussian-splatting",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A PyTorch implementation of Gaussian Splatting for 3D Scene Reconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gaussian-splatting",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            line.strip()
            for line in open("tests/requirements-test.txt")
            if line.strip() and not line.startswith("#")
        ],
        "cuda": ["cupy-cuda11x>=12.0.0"],  # Replace 11x with your CUDA version
    },
    entry_points={
        "console_scripts": [
            "gaussian-splatting=gaussian_splatting.main:main",
        ],
    },
) 