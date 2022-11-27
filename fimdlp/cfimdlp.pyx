# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector

cdef extern from "CPPFImdlp.h" namespace "CPPFImdlp":
    cdef cppclass CPPFImdlp:
        CPPFImdlp() except + 
        vector[double] cutPoints(vector[float]&, vector[int]&)

cdef class CFImdlp:
    cdef CPPFImdlp *thisptr
    def __cinit__(self):
        self.thisptr = new CPPFImdlp()
    def __dealloc__(self):
        del self.thisptr
    def cut_points(self, X, y):
        return self.thisptr.cutPoints(X, y)
