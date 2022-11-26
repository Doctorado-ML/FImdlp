"""
    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.
"""

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="fimdlp",
            sources=["cfimdlp.pyx", "FImdlp.cpp"],  
            language="c++",  
        ),
    ]
)
