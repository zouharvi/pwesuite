from setuptools import setup, find_packages

setup(
    name='PWESuite',
    version='0.0.0',
    url='https://github.com/cuichenx/phonetic-representation',
    author='CMU & ETH Affiliates',
    author_email='TODO',
    description='Evaluation suite for phonetic word embeddings and provides a few models.',
    packages=find_packages(),
    # the requirements here should only be for the evaluation, so e.g. transformers should not be here
    install_requires=[
        "panphon2>=0.3",
        "numpy>=1.21",
        "transformers>=4.22",
        "wandb>=0.13",
        "panphon>=0.20",
        "epitran>=1.20",
        "tqdm>=4.64",
        "scikit-learn>=1.1",
        "scipy>=1.7",
        "emoji>=2.2",
        "python-Levenshtein>=0.12",
    ],
)
