# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "CPPFImdlp.h" namespace "CPPFImdlp":
    cdef cppclass CPPFImdlp:
        CPPFImdlp() except + 
        CPPFImdlp(int, bool) except + 
        vector[float] cutPoints(vector[float]&, vector[int]&)

cdef class CFImdlp:
    cdef CPPFImdlp *thisptr
    def __cinit__(self, precision=6, debug=False):
        self.thisptr = new CPPFImdlp(precision, debug)
    def __dealloc__(self):
        del self.thisptr
    def cut_points(self, X, y):
        return self.thisptr.cutPoints(X, y)
