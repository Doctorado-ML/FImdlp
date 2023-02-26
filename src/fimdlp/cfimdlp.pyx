# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp cimport bool
import numpy as np

cdef extern from "limits.h":
    cdef int INT_MAX
cdef extern from "../cppmdlp/CPPFImdlp.h" namespace "mdlp":
    ctypedef float precision_t
    cdef cppclass CPPFImdlp:
        CPPFImdlp() except + 
        CPPFImdlp(size_t, int) except + 
        CPPFImdlp& fit(vector[precision_t]&, vector[int]&)
        int get_depth()
        vector[precision_t] getCutPoints()
        string version()
        
cdef class CFImdlp:
    cdef CPPFImdlp *thisptr
    def __cinit__(self, size_t min_length=3, int max_depth=INT_MAX):
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

cdef extern from "ArffFiles.h":
    cdef cppclass ArffFiles:
        ArffFiles() except +
        void load(string, bool)
        unsigned long int getSize()
        string getClassName()
        string getClassType()
        string trim(const string&)
        vector[vector[float]]& getX()
        vector[int]& getY()
        vector[string] getLines()
        vector[pair[string, string]] getAttributes()

cdef class CArffFiles:
    cdef ArffFiles *thisptr
    def __cinit__(self):
        self.thisptr = new ArffFiles()
    def __dealloc__(self):
        del self.thisptr
    def load(self, string filename, bool verbose = True):
        self.thisptr.load(filename, verbose)
    def get_size(self):
        return self.thisptr.getSize()
    def get_class_name(self):
        return self.thisptr.getClassName()
    def get_class_type(self):
        return self.thisptr.getClassType()
    def get_X(self):
        return np.array(self.thisptr.getX()).T
    def get_y(self):
        return self.thisptr.getY()
    def get_lines(self):
        return self.thisptr.getLines()
    def get_attributes(self):
        return self.thisptr.getAttributes()
    def __reduce__(self):
        return (CArffFiles, ())
   