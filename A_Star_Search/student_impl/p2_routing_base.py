import os
import time
from functools import total_ordering
### Please do use this PriorityQueue implementation
from queue import PriorityQueue
from typing import Any, List, Optional, Tuple, Union
try:
    import matplotlib.pyplot as plt
except:
    print("No matpotlib installed")
import numpy as np

#############################################
# This is the base class for A Star Search
# Please do not change any code in this file
#############################################

__all__ = ["GridAstarNode", "A_Star_Search_Base", "PriorityQueue"]


@total_ordering
class GridAstarNode:
    def __init__(
        self,
        pos: Tuple[int] = (0, 0),
        cost_g: Union[float, int] = float("inf"),
        cost_f: Union[float, int] = float("inf"),
        bend_count: Union[float, int] = float("inf"),
        visited: bool = False,
        parent: Optional[object] = None,
        neighbors: List[object] = [],
    ) -> None:
        self.pos = tuple(pos)
        self.cost_g = cost_g
        self.cost_f = cost_f
        self.bend_count = bend_count
        self.visited = visited
        self.parent = parent
        self.neighbors = neighbors
    ### This __eq__ and __lt__ function is to make the Node object comparable, so the priority queue knows how to sort nodes.
    ### A node with smaller cost_f will have higher priority.
    ### If two nodes have the same cost_f, a node with smaller bend_count will have higher priotity
    ### If two nodes have the same cost_f and same bend_count. The one pushed into the queue first will have higher priority.
    ### You do not need to explicitly call __eq__ or __lt__. The priority queue will handle the 'sorting' internally. You just need to push a GridAstarNode into the queue 'queue.put(node)', and pop the lowest cost node out of the queue: 'node = queue.get()'.
    def __eq__(self, other):
        return (self.cost_f, self.bend_count) == (other.cost_f, other.bend_count)

    def __lt__(self, other):
        return (self.cost_f, self.bend_count) < (other.cost_f, other.bend_count)


class A_Star_Search_Base(object):
    def __init__(self) -> None:
        super().__init__()
        self.grid_size = [0, 0]
        self.n_pins = 0
        self.n_blockages = 0
        (
            self.pin_pos_x,
            self.pin_pos_y,
            self.blockage_pos_x,
            self.blockage_pos_y,
            self.blockage_size_x,
            self.blockage_size_y,
        ) = [None] * 6

    # Please do not override the method
    def read_benchmark(self, file_path: str):
        """Read routing benchmark from file_path

        Args:
            file_path (str): path to the graph description file

        Returns:
            Tuple[List[List[int]], int, int]: array of nets (net to node map), number of nets, number of nodes
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
            self.grid_size = [int(i) for i in lines[0].strip().split(" ")]
            self.n_pins, self.n_blockages = [int(i) for i in lines[1].strip().split(" ")]
            pin_pos = np.array(
                [[int(j) for j in i.strip().split(" ")[1:]] for i in lines[2 : 2 + self.n_pins]]
            )
            self.pin_pos_x, self.pin_pos_y = pin_pos[..., 0], pin_pos[..., 1]
            blockages = np.array(
                [
                    [int(j) for j in i.strip().split(" ")[1:]]
                    for i in lines[2 + self.n_pins : 2 + self.n_pins + self.n_blockages]
                ]
            )
            self.blockage_pos_x = blockages[..., 0]
            self.blockage_pos_y = blockages[..., 1]
            self.blockage_size_x = blockages[..., 2]
            self.blockage_size_y = blockages[..., 3]
            # print(self.grid_size, self.n_pins, self.n_blockages)
        self.blockage_map = np.zeros([self.grid_size[1], self.grid_size[0]], dtype=np.bool)
        for x, y, w, h in zip(
            self.blockage_pos_x, self.blockage_pos_y, self.blockage_size_x, self.blockage_size_y
        ):
            self.blockage_map[y : y + h, x : x + w] = 1
        return (
            self.grid_size,
            self.n_pins,
            self.n_blockages,
            self.pin_pos_x,
            self.pin_pos_y,
            self.blockage_pos_x,
            self.blockage_pos_y,
            self.blockage_size_x,
            self.blockage_size_y,
        )

    def initialize(self):
        """Initialize necessary data structures before starting solving the problem"""
        raise NotImplementedError

    # Please do not override this method
    def _has_bend(self, node: GridAstarNode, neighbor: GridAstarNode) -> bool:
        ### If three adjacent nodes have the same x coordinates or have the same y coordinates, they are on the same line, then there is no bend. Otherwise, those three nodes form a bend.
        ### Three nodes mean: node.parent -> node -> neighbor
        if node.parent is not None:
            parent_pos = node.parent.pos
            node_pos = node.pos
            neighbor_pos = neighbor.pos
            if not (node_pos[0] == parent_pos[0] == neighbor_pos[0]) and not (
                node_pos[1] == parent_pos[1] == neighbor_pos[1]
            ):
                return True
        return False

    # Please do not override this method
    def _find_nearest_target_dist(self, srcs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """heuristic cost based on Manhatten Distance. Support parallel evaluation using tensor operations.

        Args:
            srcs (np.ndarray): source node positions of shape [#srcs, 2]
            targets (np.ndarray): target node positions of shape [#tar, 2]

        Returns:
            np.ndarray: heuristic cost for each source nodes of shape [#srcs]
        """
        srcs = srcs[:, np.newaxis, :]  # [#srcs, 1, 2]
        targets = targets[np.newaxis, ...]  # [1, #tar, 2]
        dist = np.abs(srcs - targets).sum(-1)  # [#srcs, #tar]
        nearest_dist = np.min(dist, axis=1)
        return nearest_dist  # [#tar]

    # Please do not override this method
    def _find_manhattan_dist_to_target(self, src: Tuple[int, int], target: Tuple[int,int]) -> int:
        ### Since only 2-pin net is used, you just need to pass the location of a source and a target to this function, and it will return a Manhattan distance. This is easier to use than the _find_nearest_target_dist function.
        return abs(src[0] - target[0]) + abs(src[1] - target[1])

    # Please do not override this method
    def _backtrack(self, node: GridAstarNode) -> List[Tuple[int]]:
        """Backtrack from the target node to the source node to generate point-wise path

        Args:
            node (GridAstarNode): target node

        Returns:
            List[Tuple[int]]: point-wise path from source node to target node. Each point is the node position.
        """
        # include root -> 1 -> 2 ->... -> node
        path = [node.pos]
        while node.parent is not None:
            path.append(node.parent.pos)
            node = node.parent
        return path[::-1]

    # Please do not override this method
    def _merge_path(self, path: List[Tuple[int]]) -> List[Tuple[Tuple[int, int]]]:
        """Extract vector-based path representation from point-wise path
            e.g.,
            from 8 points
                [(1,1), (1,2), (1,3), (1,4), (2,4), (3,4), (3,5), (3,6)]
                   |-------------------->
                                        |------------->
                                                      |------------->
            to 3 vectors
                [((1,1),(1,4)), ((1,4),(3,4)), ((3,4),(3,6))]

        Args:
            path (List[Tuple[int]]): A point-wise path containing a list of node pos.

        Returns:
            List[Tuple[Tuple[int]]]: A vector-wise path containing a list of (src, dst) position pairs
        """
        vec: List = []
        if len(path) == 1:
            vec.append((path[0], path[0]))
            return vec
        if len(path) == 2:
            vec.append((path[0], path[1]))
            return vec
        i, j, k = 0, 1, 2
        for k in range(2, len(path)):
            u = path[i]
            v1 = path[j]
            v2 = path[k]
            if u[0] == v1[0] == v2[0] or u[1] == v1[1] == v2[1]:
                j = k
            else:
                vec.append((u, v1))
                i = j
                j = k
        vec.append((path[i], path[j]))
        return vec

    # Please do not override this method
    def _split_path(self, path: List[Tuple[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        # change from vector-wise path to point-wise path
        site = np.zeros(self.grid_size, dtype=np.bool)
        for u, v in path:
            xl, xh = min(u[0], v[0]), max(u[0], v[0])
            yl, yh = min(u[1], v[1]), max(u[1], v[1])
            site[xl : xh + 1, yl : yh + 1] = 1
        path_x, path_y = np.nonzero(site)
        path = list(zip(path_x.tolist(), path_y.tolist()))
        return path

    def route_one_net(
        self,
    ) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], int, List[int], List[int]]:
        """route one multi-pin net using the A star search algorithm

        Return:
            path (List[Tuple[Tuple[int], Tuple[int]]]): the vector-wise routing path described by a list of (src, dst) position pairs
            wl (int): total wirelength of the routing path
            wl_list (List[int]): a list of wirelength of each routing path
            n_visited_list (List[int]): the number of visited nodes in the grid in each iteration
        """
        raise NotImplementedError

    # Please do not override the method
    def solve(self) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], int, List[int], List[int]]:
        self.initialize()
        return self.route_one_net()

    def plot_solution(
        self,
        sol: Tuple[List[Tuple[Tuple[int], Tuple[int]]], int, List[int], List[int]],
        filepath: str = "solution_vis.png",
        node_list: List[GridAstarNode]=None,
    ):
        path, wl = sol[0], sol[1]
        grid = self.blockage_map.copy().astype(np.int32)
        # grid[path[..., 1], path[..., 0]] = -1
        for s, t in path:
            xl, xh = min(s[0], t[0]), max(s[0], t[0])
            yl, yh = min(s[1], t[1]), max(s[1], t[1])
            grid[yl : yh + 1, xl : xh + 1] = -1

        grid[self.pin_pos_y[0], self.pin_pos_x[0]] = 3
        grid[self.pin_pos_y[1:], self.pin_pos_x[1:]] = 2
        plt.imshow(grid, vmin=-1, vmax=3, cmap="RdBu_r")
        ax = plt.gca()
        ax.set_xticks(np.arange(0, len(self.blockage_map), 1))
        ax.set_yticks(np.arange(0, len(self.blockage_map), 1))
        ax.set_xticks(np.arange(-.5, len(self.blockage_map), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(self.blockage_map), 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        if node_list is not None:
            # print(node_list)
            for node in node_list:
                plt.annotate(f"f:{node.cost_f:d}\ng:{node.cost_g:d},b:{node.bend_count:d}\n{node.parent.pos if node.parent is not None else ''}", xy=(node.pos[0]-0.45, node.pos[1]+0.3), fontsize=3)
        plt.title(os.path.basename(filepath)[:-4] + f" WL: {wl}")
        plt.savefig(filepath, dpi=300)

    # Please do not override the method
    def verify_solution(self, path: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> bool:
        path = set(self._split_path(path))
        if not all((x, y) in path for x, y in zip(self.pin_pos_x, self.pin_pos_y)):
            print(f"Not all pins are connected")
            return False, "NOT_ALL_PINS_CONNECTED"
        if any(self.blockage_map[y, x] for (x, y) in path):
            print(f"Path overlapped with blockages")
            return False, "PATH_ON_BLOCKAGES"
        if not all(0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] for (x, y) in path):
            print(f"Path outside grid")
            return False, "PATH_OUTSIDE_GRID"
        return True, "PASSED"

    # Please do not override the method
    def profile(self, n_runs: int = 10) -> Tuple[float, float]:
        runtime = 0
        for _ in range(n_runs):
            start = time.time()
            self.solve()
            end = time.time()
            runtime += end - start
        runtime /= n_runs

        # start_mem = mp.memory_usage(max_usage=True)
        # res = mp.memory_usage(proc=(self.solve, []), max_usage=True, retval=True)
        # max_mem = res[0]
        # used_mem = max_mem - start_mem
        used_mem = 0  # no mem profile for now
        return runtime, used_mem

    # Please do not override the method
    def dump_output_file(
        self,
        path: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        wl: int,
        wl_list: List[int],
        n_visited_list: int,
        runtime: float,
        used_mem: float,
        output_path: str,
    ) -> None:
        path = np.array(path).reshape(-1, 4).astype(str).tolist()
        n_vec = len(path)
        with open(output_path, "w") as f:
            output = (
                str(n_vec)
                + "\n"
                + "\n".join([" ".join(v) for v in path])
                + "\n"
                + str(wl)
                + "\n"
                + " ".join(map(str, wl_list))
                + "\n"
                + " ".join(map(str, n_visited_list))
                + "\n"
                + str(runtime)
                + "\n"
                + str(used_mem)
            )
            f.write(output)

    # Please do not override the method
    def load_solution(self, output_path: str):
        with open(output_path, "r") as f:
            lines = f.readlines()
            n_vec = int(lines[0].strip())
            path = []
            for i in lines[1 : 1 + n_vec]:
                i = i.strip().split(" ")
                path.append(((int(i[0]), int(i[1])), (int(i[2]), int(i[3]))))
            wl = int(lines[1 + n_vec].strip())
            wl_list = [int(i) for i in lines[2 + n_vec].strip().split(" ")]
            n_visited_list = [int(i) for i in lines[3 + n_vec].strip().split(" ")]
            runtime = float(lines[4 + n_vec].strip())
            used_mem = float(lines[5 + n_vec].strip())
            # print(path, wl, wl_list, n_visited_list)
        return path, wl, wl_list, n_visited_list, runtime, used_mem
