# distutils: language = c++
import cython
import numpy as np
from libcpp.vector cimport vector

# keeps information about pixel (x, y)
cdef struct Node:
    int x
    int y
    int active
    int expanded

    int total_cost
    int has_infinite_cost

    Node* next
    Node* prev

cdef struct List:
    int size
    Node* tail

cdef void list_push(Node* node, List* lst):
    if lst[0].tail != NULL:
        lst[0].tai