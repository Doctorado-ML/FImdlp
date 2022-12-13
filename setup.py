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
                "src/cppmdlp/CPPFImdlp.cpp",
                "src/cppmdlp/Metrics.cpp",
            ],
            language="c++",
            include_dirs=["fimdlp"],
            extra_compile_args=["-std=c++2a"],
        ),
    ]
)
