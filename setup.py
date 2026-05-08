from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dhaka-crime-classification",
    version="1.0.0",
    author="Sarder Junaid Ahmed",
    author_email="your.email@example.com",
    description="Fairness-Aware ML for Crime Classification in Dhaka Metropolitan Area",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dhaka-crime-classification",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
