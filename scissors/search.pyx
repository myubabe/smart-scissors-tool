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
        lst[0].tail[0].next = node
        node[0].prev = lst[0].tail

    lst[0].tail = node
    node[0].active = True
    lst[0].size+=1

cdef void list_remove_node(Node* node, List* lst):
    if node == lst[0].tail:
        if node[0].prev != NULL:
            node[0].prev[0].next = NULL
            lst[0].tail = node[0].prev
        else:
            lst[0].tail = NULL

    elif node[0].prev == NULL:
        node[0].next[0].prev = NULL
    else:
        node[0].prev[0].next = node[0].next
        node[0].next[0].prev = node[0].prev

    node[0].next = NULL
    node[0].prev = NULL

    node[0].active = False
    lst[0].size-=1

cdef Node * list_pop(List* lst):
    cdef Node* tail = lst[0].tail
    list_remove_node(tail, lst)
    return tail

cdef Node* get_node_ptr(int x, int y, vector[vector[Node]]* storage):
    cdef Node* node = &storage[0][x][y]
    node[0].x = x