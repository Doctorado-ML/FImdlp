# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "limits.h":
    cdef int INT_MAX
cdef extern from "../cppmdlp/CPPFImdlp.h" namespace "mdlp":
    ctypedef float precision_t
    cdef cppclass CPPFImdlp:
        CPPFImdlp() except + 
        CPPFImdlp(int, int) except + 
        CPPFImdlp& fit(vector[precision_t]&, vector[int]&)
        int get_depth()
        vector[precision_t] getCutPoints()
        string version()
        
cdef class CFImdlp:
    cdef CPPFImdlp *thisptr
    def __cinit__(self, int min_length=3, int max_depth=INT_MAX):
        self.thisptr = new CPPFImdlp(min_length, max_depth)
    def __dealloc__(self):
        del self.thisptr
    def fit(self, X, y):
        self.thisptr.fit(X, y)
        return self
    def get_cut_points(self):
        return self.thisptr.getCutPoints()
    def get_version(self):
        return self.thisptr.version()
    def get_depth(self):
        return self.thisptr.get_depth()
    def __reduce__(self):
        return (CFImdlp, ())

cdef extern from "Factorize.h" namespace "utils":
    vector[int] cppFactorize(vector[string] &input_vector)
def factorize(input_vector):
    return cppFactorize(input_vector)