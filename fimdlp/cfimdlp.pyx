# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "CPPFImdlp.h" namespace "CPPFImdlp":
    cdef cppclass CPPFImdlp:
        CPPFImdlp() except + 
        CPPFImdlp(int, bool) except + 
        vector[float] cutPointsAnt(vector[float]&, vector[int]&)
        void debugPoints(vector[float]&, vector[int]&)


cdef class CFImdlp:
    cdef CPPFImdlp *thisptr
    def __cinit__(self, precision=6, debug=False):
        self.thisptr = new CPPFImdlp(precision, debug)
    def __dealloc__(self):
        del self.thisptr
    def cut_points_ant(self, X, y):
        return self.thisptr.cutPointsAnt(X, y)
    def debug_points(self, X, y):
        return self.thisptr.debugPoints(X, y)
