
import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sps
DTYPE_float = np.float32
DTYPE_int = np.int32
ctypedef np.float32_t DTYPE_float_t
ctypedef np.int32_t DTYPE_int_t

cdef class FactorFunction:
    cpdef double get_value(self, int y1, int y2, np.ndarray[DTYPE_float_t, ndim=1] weight) except? -1

cdef class EdgeFactorFunction(FactorFunction):
    cdef public int num_label,edge_type,num_edge_feature_each_type,num_attrib_parameter
    cdef public dict feature_offset
    cpdef double get_value(self, int y1, int y2, np.ndarray[DTYPE_float_t, ndim=1] weight) except? -1

cdef class DiffMax:
    cdef double diff_max




cdef class FactorGraph:
    cdef public bint labeled_given,converged
    cdef public int n,m,num_label,num_node,factor_node_used
    cdef public list var_node,factor_node, edge_type_func_list
    cdef public np.ndarray bfs_node,p_node
    cdef public DiffMax diff_max
    cdef public int all_diff_size
    cdef dict __dict__

    cpdef public add_edge_done(self)
    cpdef add_edge(self, int a, int b, int edge_type)
    cpdef clear_data_for_sum_product(self)
    cpdef set_variable_label(self, int u, int y)
    cpdef set_variable_state_factor_y(self, int u, int y, double v)
    cpdef set_variable_state_factor(self, int u, np.ndarray[DTYPE_float_t, ndim=1] state_factor)
    cpdef public gen_propagate_order(self)
    cpdef belief_propagation(self, int max_iter, np.ndarray[DTYPE_float_t, ndim=1] weight)
    cpdef calculate_marginal(self, np.ndarray[DTYPE_float_t, ndim=1] weight)
    cpdef clean(self)