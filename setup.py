from os import path
from setuptools import setup, find_packages
from codecs import open

with open(
    path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8"
) as f:
    long_description = f.read()

setup(
    name="numpy-quant",
    version="0.1.0",
    license="MIT",
    author="Tilmann E. Bartsch",
    url="https://github.com/tebartsch/numpy-quant",
    description="Quantize ONNX-models with arbitrary bit-width using only numpy.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    keywords="machine-learning quantization numpy",
    packages=find_packages(include=["numpy_quant"]),
    python_requires="~= 3.7",
    install_requires=[
        "onnx~=1.13.0",
        "numpy~=1.23.5"
    ],
    extras_require={
        "test": ["torch~=1.13.0",
                 "torchvision~=0.14.0",
                 "torchaudio~=0.13.0",
                 "scikit-learn~=1.2.0",
                 "plotext~=5.2.8",
                 "onnxruntime~=1.13.1",
                 "transformers~=4.25.1",
                 "datasets~=2.7.1",
                 "tqdm~=4.64.1",
                 ],
    },
)