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
            include_dirs=["fimdlp"],
        ),
    ]
)

# from Cython.Build import cythonize
# setup(
#     ext_modules=cythonize(
#         Extension(
#             "fimdlp", 
#             sources=["fimdlp/cfimdlp.pyx", "fimdlp/FImdlp.cpp"],  
#             language="c++", 
#             include_dirs=["fimdlp"], 
#         ),
#         include_path=["./fimdlp"],
#     )
# )

