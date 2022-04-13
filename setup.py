import setuptools
import os

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
README = os.path.join(CUR_DIR, "README.md")
with open("README.md", "r") as fd:
    long_description = fd.read()

setuptools.setup(
    name="seq2seq-pytorch",
    version="0.1.2",
    description="Industrial-grade implementation of seq2seq algorithm based on Pytorch, integrated beam search algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chiang97912/seq2seq",
    author="Chiang97912",
    author_email="chiang97912@gmail.com",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchtext>=0.3.1",
        "nltk>=3.5",
        "numpy>=1.19.5",
        "nltk>=3.5",
        "jieba>=0.42.1",
    ],
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ),

    keywords='seq2seq'
)
