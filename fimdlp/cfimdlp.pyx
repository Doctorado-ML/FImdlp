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
        CPPFImdlp(int, bool) except + 
        vector[CutPointBody] getCutPoints()
        vector[float] cutPointsAnt(vector[float]&, vector[int]&)
        void debugPoints(vector[float]&, vector[int]&)
        void computeCutPoints(vector[float]&, vector[int]&)
        

class PCutPointBody:
    def __init__(self, start, end, fromValue, toValue):
        self.start = start
        self.end = end
        self.fromValue = fromValue
        self.toValue = toValue

cdef class CFImdlp:
    cdef CPPFImdlp *thisptr
    def __cinit__(self, precision=6, debug=False):
        self.thisptr = new CPPFImdlp(precision, debug)
    def __dealloc__(self):
        del self.thisptr
    def cut_points(self, X, y):
        self.thisptr.computeCutPoints(X, y)
        return  self.thisptr.getCutPoints()
    def cut_points_ant(self, X, y):
        return self.get_cut_points(X, y)
    def debug_points(self, X, y):
        return self.thisptr.debugPoints(X, y)
 