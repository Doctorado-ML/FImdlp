# FImdlp
[![CI](https://github.com/Doctorado-ML/FImdlp/actions/workflows/main.yml/badge.svg)](https://github.com/Doctorado-ML/FImdlp/actions/workflows/main.yml)
[![CodeQL](https://github.com/Doctorado-ML/FImdlp/actions/workflows/codeql.yml/badge.svg)](https://github.com/Doctorado-ML/FImdlp/actions/workflows/codeql.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/8b4d784fee13401588aa8c06532a2f6d)](https://www.codacy.com/gh/Doctorado-ML/FImdlp/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Doctorado-ML/FImdlp&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/Doctorado-ML/FImdlp/branch/main/graph/badge.svg?token=W8I45B5Z3J)](https://codecov.io/gh/Doctorado-ML/FImdlp)
[![pypy](https://img.shields.io/pypi/v/FImdlp?color=g)](https://img.shields.io/pypi/v/FImdlp?color=g)
![https://img.shields.io/badge/python-3.9%2B-blue](https://img.shields.io/badge/python-3.9%2B-brightgreen)

Discretization algorithm based on the paper by Usama M. Fayyad and Keki B. Irani 


Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning. In Proceedings of the 13th International Joint Conference on Artificial Intelligence (IJCAI-95), pages 1022-1027, Montreal, Canada, August 1995.


## Installation

```bash
git clone --recurse-submodules https://github.com/doctorado-ml/FImdlp.git
```

## Build and usage sample

### Python sample

```bash
pip install -e .
python samples/sample.py iris  
python samples/sample.py iris --alternative
python samples/sample.py -h # for more options
```

### C++ sample

```bash
cd samples
mkdir build
cd build
cmake ..
make
./sample iris
```
