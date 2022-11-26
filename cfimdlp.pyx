# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
cdef extern from "FImdlp.h" namespace "FImdlp":
    cdef cppclass FImdlp:
        FImdlp() except + 
        vector[float] cutPoints(vector[int]&, vector[int]&)

cdef class CFImdlp:
    cdef FImdlp *thisptr
    def __cinit__(self):
        self.thisptr = new FImdlp()
    def __dealloc__(self):
        del self.thisptr
    def cut_points(self, X, y):
        return self.thisptr.cutPoints(X, y)
