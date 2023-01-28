from setuptools import setup, find_packages

setup(
    name='PhoneticRepresentation',
    version='0.0.0',
    url='https://github.com/cuichenx/phonetic-representation',
    author='CMU & ETH Affiliates',
    author_email='TODO',
    description='Evaluation suite for phonetic word embeddings and a distance learning model.',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "transformers>=4.22",
        "wandb>=0.13",
        "panphon>=0.20",
        "epitran>=1.20",
        "tqdm>=4.64",
        "scikit-learn>=1.1",
        "scipy>=1.7",
    ],
)
