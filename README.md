# FImdlp
[![CI](https://github.com/Doctorado-ML/FImdlp/actions/workflows/main.yml/badge.svg)](https://github.com/Doctorado-ML/FImdlp/actions/workflows/main.yml)
[![CodeQL](https://github.com/Doctorado-ML/FImdlp/actions/workflows/codeql.yml/badge.svg)](https://github.com/Doctorado-ML/FImdlp/actions/workflows/codeql.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/8b4d784fee13401588aa8c06532a2f6d)](https://app.codacy.com/gh/Doctorado-ML/FImdlp/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![codecov](https://codecov.io/gh/Doctorado-ML/FImdlp/branch/main/graph/badge.svg?token=W8I45B5Z3J)](https://codecov.io/gh/Doctorado-ML/FImdlp)
[![pypy](https://img.shields.io/pypi/v/FImdlp?color=g)](https://pypi.org/project/FImdlp)
![https://img.shields.io/badge/python-3.11%2B-blue](https://img.shields.io/badge/python-3.11%2B-brightgreen)

Discretization algorithm based on the paper by Usama M. Fayyad and Keki B. Irani

Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning. In Proceedings of the 13th International Joint Conference on Artificial Intelligence (IJCAI-95), pages 1022-1027, Montreal, Canada, August 1995.

## Installation

### From PyPI

```bash
pip install FImdlp
```

### From source (development)

The project no longer relies on a git submodule — the C++ sources live under `src/cpp/`. A regular clone is enough:

```bash
git clone https://github.com/doctorado-ml/FImdlp.git
cd FImdlp
make deps      # install the [dev] extras (build, twine, pip-audit, black, flake8, coverage)
make install   # editable install; compiles the Cython/C++ extension in place
```

Run `make help` to list every available target.

## Quick start

```python
from sklearn.datasets import load_iris
from fimdlp.mdlp import FImdlp

X, y = load_iris(return_X_y=True)
clf = FImdlp().fit(X, y)

# Discretize
X_disc = clf.transform(X)

# Inspect cut points: [vmin, c1, ..., cn, vmax] per feature
for f, cuts in enumerate(clf.get_cut_points()):
    print(f"feature {f}: {cuts}")
```

Constructor parameters:

| Parameter | Default | Description |
|---|---|---|
| `n_jobs` | `-1` | Threads for per-feature `fit`/`transform`. `-1` uses all cores. |
| `min_length` | `3` | Minimum samples in an interval to consider further splits. |
| `max_depth` | `1e6` | Maximum recursion depth of the splitting procedure. |
| `max_cuts` | `0` | Cap on intermediate cut points per feature (`0` = unlimited; `<1` is interpreted as a fraction of samples). |

## Make targets

| Target | What it does |
|---|---|
| `make help` | List every target. |
| `make deps` | Install the `[dev]` extras (build, twine, pip-audit, black, flake8, coverage). |
| `make install` | Editable install (`pip install -e .`); rebuilds the C++/Cython extension. |
| `make test` | Run unit tests with coverage. Rebuilds the extension if the `.so` is missing. |
| `make coverage` | Run tests then print the coverage report. |
| `make lint` | Format with `black` and lint with `flake8`. |
| `make build` | Produce wheel + sdist in `dist/`. |
| `make publish` | `make build` + `twine check` + `twine upload`. |
| `make audit` | Run `pip-audit` on the installed packages. |
| `make sample_py` | Run the Python sample on the iris dataset. |
| `make sample_cpp` | Build and run the C++ sample on the iris dataset. |
| `make version` | Show Python, FImdlp and bundled mdlp versions. |
| `make clean` | Remove build artifacts, caches and the compiled extension. |

## Running the samples

### Python sample

```bash
make sample_py
# equivalent to:
#   cd samples && python sample.py iris
```

Other options:

```bash
python samples/sample.py iris            # default settings
python samples/sample.py iris -c 2       # cap intermediate cut points to 2
python samples/sample.py iris -m 3       # cap recursion depth to 3
python samples/sample.py iris -n 25      # set min_length to 25
python samples/sample.py -h              # full option list
```

### C++ sample

```bash
make sample_cpp
# equivalent to:
#   cd samples && cmake -B build -S . && cmake --build build && cd build && ./sample -f iris
```

Other options:

```bash
cd samples/build
./sample -f iris -c 2     # cap intermediate cut points to 2
./sample -f glass -m 3    # change dataset and depth
./sample -h               # full option list
```

## Based on

[https://github.com/rmontanana/mdlp](https://github.com/rmontanana/mdlp)
