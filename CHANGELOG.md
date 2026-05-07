# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-07

First stable release. The C++ MDLP core has been replaced with the upstream
`mdlp 2.1.3` sources, the `transform` path now goes through C++, the
estimator is fully compatible with scikit-learn 1.8, and the package
build has been cleaned up for PyPI publishing.

### ⚠️ Breaking changes

- **`get_cut_points()` now returns `[vmin, c1, ..., cn, vmax]` per feature**
  (previously only the intermediate cut points `[c1, ..., cn]`).
  Code that fed the result into `np.searchsorted(cut_points, X)` should
  either switch to `clf.transform(X)` or slice with `cut_points[1:-1]`.
- **Python ≥ 3.11** required (was ≥ 3.9).
- **scikit-learn ≥ 1.8.0** required. The deprecated `_more_tags` hook has
  been replaced by `__sklearn_tags__`; subclasses must migrate accordingly.
- **C++17** required (was C++11).
- The `cppmdlp` git submodule has been removed; the C++ sources are now
  vendored in `src/cpp/`. Existing checkouts should be re-cloned (no
  `--recurse-submodules` needed).

### Added

- `FImdlp.transform` and `CFImdlp.transform` are now backed by the C++
  `Discretizer::transform` (using `upper_bound` over the intermediate
  cuts), instead of the previous Python-side `np.searchsorted` path.
- Lazy cut-point loading: `get_cut_points()` and `get_states_feature()`
  fetch from the C++ object on first access and cache the result;
  `join_fit` invalidates the cache for the re-fitted feature.
- `CFImdlp` is now properly pickleable: `__reduce__` persists the
  constructor args, the cut points and the recursion depth, and a helper
  rebuilds the state on unpickle. Required for
  `sklearn.utils.estimator_checks.check_estimator`.
- New Make targets: `deps`, `publish`, `sample_py`, `sample_cpp`.
  `make help` prints the full list.
- Optional dependency group `[dev]` (`build`, `twine`, `pip-audit`,
  `black`, `flake8`, `coverage`).
- Sdist (`*.tar.gz`) is now produced alongside the wheel.
- Seven new tests covering the C++ transform path, sentinel exposure,
  lazy cache semantics, cache invalidation on `join_fit`, deterministic
  re-`transform`, out-of-range value clamping and state/cut consistency.
- `iris.arff` shipped with the project tests at
  `src/fimdlp/tests/datasets/iris.arff`.

### Changed

- C++ MDLP implementation upgraded from the previous bundled 1.1.2 to
  upstream **mdlp 2.1.3**, picking up several upstream fixes:
  - Entropy computation no longer assumes contiguous label values
    `[0..K-1]` (previous version could write out of bounds for sparse or
    high-valued labels).
  - `Discretizer::transform` clears its output buffer on every call;
    repeated `transform()` invocations are now deterministic.
  - `valueCutPoint` guards against `size_t` underflow when duplicate
    values reach the interval boundary.
  - `resizeCutPoints` checks `indices` bounds before access.
  - `Metrics` cache access is protected by a `std::mutex`.
- `CFImdlp.fit`/`transform` accept any numeric numpy array (cast to
  `float32` / `int32` internally) instead of requiring pre-converted
  Python lists.
- `FImdlp.fit`/`transform` use `sklearn.utils.validation.validate_data`,
  which automatically tracks `n_features_in_` and emits the canonical
  sklearn error messages on shape mismatch.
- All C++ sources live in `src/cpp/` (`CPPFImdlp.{h,cpp}`,
  `Metrics.{h,cpp}`, `typesFImdlp.h`, plus the relocated
  `Factorize.{h,cpp}` and `ArffFiles.{h,cpp}`).
- Build configuration:
  - `pyproject.toml` now declares packages explicitly via
    `[tool.setuptools.packages.find]` (fixes the previously-leaked
    top-level `cpp` package in published wheels).
  - Wheel excludes `*.pyx`, `*.cpp`, `*.h` and the test suite; sdist
    keeps everything required to rebuild.
  - `[build-system]` requires pinned to `setuptools>=64`, `cython>=3.0`
    (dropped redundant `wheel`).
  - PyPI classifier bumped to `Development Status :: 5 - Production/Stable`.
- `make build` no longer wipes the editable extension; `make test`
  rebuilds the extension automatically if the `.so` is missing.
- README rewritten: PyPI install instructions, dev workflow, full Make
  target table, Python and C++ sample usage with options.
- CI: dropped Windows from the test matrix; CodeQL action upgraded to
  v4; Python version matrix aligned with the new `requires-python`.

### Fixed

- `transform` now validates input shape against the fitted
  `n_features_in_` and emits the standard sklearn error message.
- `check_estimator(FImdlp())` from scikit-learn 1.8 passes (pickling,
  tags API, validation pipeline).

### Removed

- `cppmdlp` git submodule and `.gitmodules` entry.
- `_more_tags` hook (replaced by `__sklearn_tags__`).
- `make submodule` target.
- Build dependency on `wheel`.

## [0.9.4] - 2023-04-25

Last release with the C++ core consumed via `cppmdlp` submodule (mdlp 1.1.2).
