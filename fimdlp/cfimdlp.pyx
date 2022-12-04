# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "CPPFImdlp.h" namespace "mdlp":
    cdef struct CutPointBody:
        size_t start, end;
        int classNumber;
        float fromValue, toValue;
    cdef cppclass CPPFImdlp:
        CPPFImdlp() except + 
        CPPFImdlp(bool, int, bool) except + 
        CPPFImdlp& fit(vector[float]&, vector[int]&)
        vector[int] getDiscretizedValues()
        vector[float] getCutPoints()
        

class PcutPoint_t:
    def __init__(self, start, end, fromValue, toValue):
        self.start = start
        self.end = end
        self.fromValue = fromValue
        self.toValue = toValue

cdef class CFImdlp:
    cdef CPPFImdlp *thisptr
    def __cinit__(self, precision=6, debug=False, proposal=True):
        # Proposal or original algorithm
        self.thisptr = new CPPFImdlp(proposal, precision, debug)
    def __dealloc__(self):
        del self.thisptr
    def fit(self, X, y):
        self.thisptr.fit(X, y)
        return self
    def get_discretized_values(self):
        return self.thisptr.getDiscretizedValues()
    def get_cut_points(self):
        return self.thisptr.getCutPoints()
 