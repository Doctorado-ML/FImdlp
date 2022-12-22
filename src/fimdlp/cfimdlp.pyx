# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "../cppmdlp/CPPFImdlp.h" namespace "mdlp":
    ctypedef float precision_t
    cdef cppclass CPPFImdlp:
        CPPFImdlp(int) except + 
        CPPFImdlp& fit(vector[precision_t]&, vector[int]&)
        vector[precision_t] getCutPoints()
        string version()
        
cdef class CFImdlp:
    cdef CPPFImdlp *thisptr
    def __cinit__(self, algorithm):
        self.thisptr = new CPPFImdlp(algorithm)
    def __dealloc__(self):
        del self.thisptr
    def fit(self, X, y):
        self.thisptr.fit(X, y)
        return self
    def get_cut_points(self):
        return self.thisptr.getCutPoints()
    def get_version(self):
        return self.thisptr.version()
