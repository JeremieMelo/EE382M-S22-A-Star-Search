'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-07 16:42:12
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-08 16:04:28
'''
#######################################################################
# Implementation of A Star Search
# You need to implement initialize() and route_one_net()
# All codes should be inside A Star Search class
# Name:
# UT EID:
#######################################################################

from typing import List, Tuple

import numpy as np

from .p2_routing_base import A_Star_Search_Base, GridAstarNode, PriorityQueue

__all__ = ["A_Star_Search"]

class A_Star_Search(A_Star_Search_Base):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self):
        """Initialize necessary data structures before starting solving the problem
        """
        # TODO initialize any auxiliary data structure you need
        raise NotImplementedError

    def route_one_net(self) -> Tuple[List[Tuple[Tuple[int], Tuple[int]]], int, List[int], List[int]]:
        """route one multi-pin net using the A star search algorithm

        Return:
            path (List[Tuple[Tuple[int], Tuple[int]]]): the vector-wise routing path described by a list of (src, dst) position pairs
            wl (int): total wirelength of the routing path
            wl_list (List[int]): a list of wirelength of each routing path
            n_visited_list (List[int]): the number of visited nodes in the grid in each iteration
        """
        # TODO implement your A star search algorithm for one multi-pin net.
        # To make this method clean, you can extract subroutines as methods of this class
        # But do not override methods in the parent class
        # Please strictly follow the return type requirement.

        raise NotImplementedError
