"""
    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.
"""

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="fimdlp.cppfimdlp",
            sources=[
                "src/fimdlp/cfimdlp.pyx",
                "src/cpp/CPPFImdlp.cpp",
                "src/cpp/Metrics.cpp",
                "src/cpp/Factorize.cpp",
                "src/cpp/ArffFiles.cpp",
            ],
            language="c++",
            include_dirs=["src/cpp"],
            extra_compile_args=[
                "-std=c++17",
            ],
        ),
    ]
)
