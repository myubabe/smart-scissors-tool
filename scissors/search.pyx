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
    node[0].y = y
    return node

cdef void set_cost(Node* n, long cost):
    n[0].total_cost = cost
    n[0].has_infinite_cost = False

cdef vector[vector[Node]]* make_node_storage(int w, int h):
    return new vector[vector[Node]](w, vector[Node](h, Node(0, 0, False, False, 0, True, NULL, NULL)))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def search(long [:, :, :, :]static_cost, long [:, :, :, :] dynamic_cost,
            int w, int h, int seed_x, int seed_y, int maximum_local_cost):

    # keeps information about all pixels
    cdef vector[vector[Node]]* raw_storage = make_node_storage(w, h)
    cdef Node* seed_point = get_node_ptr(seed_x, seed_y, raw_storage)

    # seed has 0 cost
    set_cost(seed_point, 0)
    # create active list
    cdef vector[List]* active_list = new vector[List](maximum_local_cost, List(0, NULL))
    # put seed point to the first bucket
    cdef int list_index = 0
    list_push(seed_point, &active_list[0][list_index])

    # next node x and next node y, current x, current y
    cdef long [:, :, :] next_node_map = np.zeros((2, w, h), dtype=np.int)

    # tracks the number of active buckets
    cdef int num_of_active_lists = 1

    cdef long tmp_cost = 0
    cdef long last_expanded_cost = 0

    cdef Node* p = NULL
    cdef Node* q = NULL

    cdef int p_x = 0
    cdef int p_y = 0

    cdef int q_x = 0
    cdef int q_y = 0

    # shift indices of neighbors
    cdef int x_shift = 0, y_shift = 0
    # [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    cdef vector[int] y_shifts = [-1, -1, -1, 0, 0, 1, 1, 1]
    cdef vector[int] x_shifts = [-1, 0, 1, -1, 1, -1, 0, 1]

    # while there are unexpanded points
    while num_of_active_lists != 0:
        last_expanded_cost -= 1

        while True:
            last_expanded_cost += 1
            list_index = last_expanded_cost % maximum_local_cost

            if active_list[0][list_index].size != 0:
                break

        p = list_pop(&active_list[0][list_index])
        p_x = p[0].x
        p_y = p[0].y
        #p = get_node_ptr(p_x, p_y, raw_storage)

        # mark 'p' as expanded
        p[0].expanded = True
        last_expanded_cost = p[0].t