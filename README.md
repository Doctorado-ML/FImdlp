# FImdlp

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/8b4d784fee13401588aa8c06532a2f6d)](https://www.codacy.com/gh/Doctorado-ML/FImdlp/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Doctorado-ML/FImdlp&amp;utm_campaign=Badge_Grade)

Discretization algorithm based on the paper by Usama M. Fayyad and Keki B. Irani 

```
Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning. In Proceedings of the 13th International Joint Conference on Artificial Intelligence (IJCAI-95), pages 1022-1027, Montreal, Canada, August 1995.
```

## Build and usage sample

### Python sample

```bash
pip install -e .
python samples/sample.py iris --original 
python samples/sample.py iris --proposal
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
