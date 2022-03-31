# Coding Assignment 2: Multi-Pin Net Detailed Routing using A Star Search
Implement one multi-pin net detailed routing using A* search on CPU.

# Recommended Python Environment
 * Python >= 3.6.6
 * Numpy >= 1.17.0
   * Install with this command: `pip3 install numpy --user`
 * tqdm >= 4.63.0
   * Install with this command: `pip3 install tqdm --user`
 * matplotlib >= 3.5.0 (Optional, intallation might fail on ECE LRC)
   * Install with this command: `pip3 install matplotlib --user`
   * This is helpful for debugging, you might want to plot on your own computer.

# Codebase Structure
 - A_Star_Search
   * `benchmarks`: text-based benchmarks
     * e.g., example_1.txt
     ```bash
        20 20
        4 7
        p0 18 1
        p1 9 13
        p2 0 8
        p3 6 6
        b0 9 8 6 3
        b1 7 9 8 1
        b2 8 15 10 2
        b3 2 14 5 3
        b4 9 5 8 2
        b5 2 18 9 1
        b6 1 1 6 2
     ```
     * The first row contains the routing grid dimension (W x H), e.g., 20 x 20.
     * The second row contains the number of pins in the net and number of blockages, e.g., a 4-pin net and 7 rectangular blockages.
     * There are #pins rows describing the pin locations:
     ```bash
     PIN_NAME PIN_POS_X PIN_POS_Y
     ```
     E.g., `p0 18 1` describes a `pin` named `p0` whose position is (18, 1).
     * The rest #blockages rows describe the position and dimension of the rectangular blockages.
     ```bash
     BLOCKAGE_NAME POS_X_LEFT POS_Y_BOTTOM SIZE_X SIZE_Y
     ```
     E.g., `b0 9 8 6 3` describes a `6 x 3` blockage whose lower-left corner is at (9, 8).
    * `output`: directory to dump out your partition solution.
      * For a student with EID: `xxxxx`, all solution files will be dumped to `output/xxxxx/`. The solution file will have the same file name as the benchmark file, e.g., `output/xxxxx/example_1.txt`
      * `output/reference`: contains ground truth solutions to the example benchmarks given to you.
      * The format of the routing solution file is as follows,
      ```bash
        7
        18 1 7 1
        7 1 7 6
        7 6 6 6
        6 6 0 6
        0 6 0 8
        6 6 6 13
        6 13 9 13
        38
        18 9 11
        59 20 31
        0.006187280019124349
        0
      ```
      * The first row is the number of straight routing paths |P| contained in the routing path P, e.g., |P|=`7` for this benchmark.
      * The next |P| rows describes the routing wires. Each row follows,
      ```bash
       SRC_NODE_X SRC_NODE_Y TAR_NODE_X TAR_NODE_Y
      ```
      * E.g., `18 1 7 1` is a vector pointing from (18, 1) to (7, 1).
      * The next row contains the total routed wirelength `wl` corresponding to the above routing solution
      * Then, the next row contains a list of routed wirelength for each path connecting one target.
        * E.g., `18 9 11` means there are 3 paths in the 4-pin net. Their wirelengths are 18, 9, and 11, respectively.
      * Then, the next row contains a list of number of visited nodes in each iteration.
        * E.g., `59 20 31` means there are 3 iterations in the 4-pin net routing program, and each iteration visited 59, 20, and 31 nodes, respectively
      * The next row is the average runtime in second (**Important grading metric**)
      * The last row is the used memory (Can ignore this, just put a 0 there)
    * `student_impl`: directory to store algorithm implementations.
      * `__init__.py`: initialize package.
      * `p2_routing_base.py`: basic class of the A star search solver `A_Star_Search_Base`. Please do not change any code in this file.
      * `eid_YOUR_EID.py`: `A_Star_Search` class inherited from `A_Star_Search_Base`.
        * The first thing you need to do is to replace `YOUR_EID` in the file name with your lowercase UT EID, e.g., eid_xxxxx.py.
        * **You need to implement the `initialize()` and `route_one_net()` methods**.
        * Please keep all codes within the `A_Star_Search` class.
        * Please only submit this single file. Any codes outside `A_Star_Search` class or outside this file will be ignored.
        * A reasonable amount of comments are encouraged to improve the code readability.
    * `p2_routing_eval.py`: script to evaluate your partition solver on given benchmarks.
      * Below is an example to perform evaluation for `YOUR_UT_EID`, e.g., `xxxxx`, on `BENCHMARK_PATH`, e.g., `./benchmarks/example_1.txt`, with execution (`-r`), grading (`-s`), runtime profiling (`-p`), draw solution (`-d`) enabled.
      * ```bash
        A_Star_Search/> python3 p2_routing_eval.py -e YOUR_UT_EID -b BENCHMARK_PATH -r -s -p
        ```
      * Use help function for more details on how to use this script. `python3 p2_routing_eval.py -h`
      * After running this script, you can find all successfully generated solutions under `output/YOUR_UT_EID/`.
      * If grading is enabled with `-s`, there will be a `score.csv` under `output/YOUR_UT_EID` to summarize your evaluation results.
      * If drawing is enabled with `-d`, there will be a `BENCHMARK_NAME_sol.png` under `output/YOUR_UT_EID` to visualize your routing solution.

# How to Debug
* Write your codes in the `A_Star_Search` class.
* You can first use your own evaluation code to debug if you find it more convenient. Create a file `A_Star_Search/my_test.py` with the following content. Change `YOUR_UT_EID` to your UT EID.
 ```bash
    import os
    from student_impl.eid_YOUR_UT_EID import A_Star_Search
    eid = "YOUR_UT_EID"
    idx = 1
    benchmark_path = f"benchmarks/example_{idx}.txt"
    output_root = "output"
    output_root = os.path.join(output_root, eid)
    if not os.path.isdir(output_root):
        os.mkdir(output_root)

    output_path = os.path.join(output_root, os.path.basename(benchmark_path))
    solver = A_Star_Search()
    solver.read_benchmark(benchmark_path)
    solution = solver.solve()solver.plot_solution(solution, os.path.join(output_root, f"example_{idx}_sol.png"))
    profiling = solver.profile(n_runs=3) # ignore memory for now. runtime will be graded
    solver.dump_output_file(*solution, *profiling, output_path)
  ```
* Use `pdb` or print function to debug your code.
* Once you successfully dump the solution file `output/YOUR_UT_EID/example_1.txt`, compare it with the ground truth solution `output/reference/example_1.txt`.
* Use `plot_solution()` function to visualize your solution.
* You can also evaluate your code using the evaluation script.
  ```bash
  A_Star_Searc> python3 p2_routing_eval.py -e YOUR_UT_EID -b ./benchmarks/example_1.txt -r -s
  ```
* Then, you can further verify your code on other example benchmarks.
* 5 exmaple benchmarks will be provided.

# Important Notes
* For multi-pin net routing, please use the greedy heuristic introduced in the lecture, i.e., starting from one pin, find the path to the nearest target -> Set all nodes along the path as sources, and launch another A star search engine from those sources to find the nearest target. Repeat this until all pins are connected.
* Please use the given `GridAstarNode` for each routing grid node and `PriorityQueue` for the visited node selection.
* Please utilize the given methods in the base class.
  * `_find_nearest_target_dist()`: heuristic cost used in A star search. Please use this as the `h(x)` cost.
  * `_backtrack()`: backtrack the path from the target node you searched to the source node. Please use this after your program found one target node.
  * `_merge_path()`: merge point-wise path to vector representation. Please use this to convert your final point-wise path solution.
  * `_has_bend()`: when selecting which neighbor node to update first, please use this as the secondary metric to prefer path with less turns or bends. Otherwise, you will get zig-zag path, which is not preferred.
* The dumped partition solution should use the original node names in the benchmark file.
* To ensure a **deterministic** solution in this multi-source multi-target routing problem,
  * Please use the first pin, i.e., `p0`, as the starting node, which is marked as red in the solution plot.
  * Please follow this rule to loop over neighbors of a node in your program, `left -> top -> right -> bottom`. E.g., if the visited node is at (`x, y`), its 4-way neighbors are `(x-1, y) -> (x, y-1) -> (x+1, y) -> (x, y+1)`. You might need to skip neighbors if they are on top of a blockage or outside the routing grid borders.
* Subroutines can be extracted and implemented as methods of the `A_Star_Search` class. This can make your `route_one_net()` method clean and easy to read.
* Please do not override methods in `A_Star_Search_Base`

# Evaluation Metrics
* You will pass a certain test case if
* your code gives the required return values with the required types without exceptions (For python code only)
* your code successfully dumps a valid solution file to `output/YOUR_UT_EID` directory
* and
* your `path` covers the exact routing grid points as the ground truth:
  * `set(solver._split_path(path)) == set(solver._split_path(ref_path))`
* and
* your path is a valid solution that passes the `verify_solution()` (see `verify_solution()` for details):
  * `solver.verify_solution(path) == True`
* and
* your `wl` equals the reference value:
  * `wl == ref_wl`
* your wl_list satisfies the following condition (the order matters):
  * `tuple(wl_list) == tuple(ref_wl_list)`
* and
* your number of visited nodes satisfies the following condition (the order matters):
  * `tuple(n_visited_list) == tuple(ref_n_visited_list)`
* **Runtime Limit** is used for this assignment
  * Each reference solution has a reference average runtime, tested on ECE LRC Daisy, `ref_runtime`
  * Your profiled average runtime `runtime` leads to the following penalty:
    ```bash
    max_ratio = 10 # tolerate 10x slower runtime at most
    ratio = runtime / ref_runtime
    # smaller than 10x the reference runtime: no penalty
    # larger than the threshold leads to exponential score decay
    runtime_weighted_score = np.e ** (1 - max(ratio/max_ratio, 1))
    ```
    * E.g., if reference runtime is 0.3 s, your runtime is 2.9 s, the penalty term is 1, which is no penalty.
    * E.g., if your runtime is 4 s, the penalty term is 0.719.
    * Any redundant debugging, e.g., drawing, print, assertion, will significantly slow down your program. Please delete them before submission.

# What to Submit if You Use This Python Codebase
Submit the single python script file named `eid_YOUR_UT_EID.py`. Please change `YOUR_UT_EID` to your lowercase UT EID, e.g., `eid_xxxxx.py`.

# How will the TA Evaluate Submitted Python Codes
TA will do the following steps to generate your score summary.
* There are 20 hidden test benchmarks which are larger than the given examples. Same benchmarks are used for all students. All benchmarks follow the same format as the example benchmarks given to students.
* Copy your file `eid_YOUR_UT_EID.py` to `student_impl/`.
* Run the following command:
  ```bash
  A_Star_Search> python3 p2_routing_eval.py -e YOUR_UT_EID -b all -r -p -s
  ```
* The generated `score.csv` file under `output/YOUR_UT_EID` will be used in final grading.
