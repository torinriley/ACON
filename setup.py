from setuptools import setup, find_packages

setup(
    name='acon',
    version='0.1.0',
    description='Adaptive Correlation Optimization Networks (ACON): A dynamic and adaptive machine learning framework.',
    author='Your Name',
    author_email='torinriley220@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
