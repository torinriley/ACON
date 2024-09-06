from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="acon",
    version="0.1.0",
    description="Adaptive Correlation Optimization Networks (ACON): A dynamic and adaptive machine learning framework.",
    long_description=long_description,
    long_description_content_type="text/markdown", 
    url="https://github.com/torinriley/Adaptive-Correlation-Optimization-Network-ACON",
    author="Torin Riley",
    author_email="torinriley220@gmail.com",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
    ],
)
