import importlib
import os
import traceback

import numpy as np
import argparse
from tqdm import tqdm

#############################################################
# Evaluate your implementation on all benchmarks
# Run this script: A_Star_Search/> python p2_routing_eval.py
#############################################################
def evaluate(
    eid: str,
    benchmark_root: str,
    output_root: str,
    impl_package: str,
    module_name: str,
    profile: bool,
    plot: bool,
    *args,
    **kwargs,
):
    """Evaluate the implementation for one student and generate outputs

    Args:
        eid (str): UT EID
    """
    eid = eid.lower()
    output_root = os.path.join(output_root, eid)
    if not os.path.isdir(output_root):
        os.mkdir(output_root)
    if os.path.isdir(benchmark_root):
        benchmarks = [os.path.join(benchmark_root, i) for i in os.listdir(benchmark_root)]
    elif os.path.isfile(benchmark_root):
        benchmarks = [benchmark_root]
    else:
        raise ValueError(f"Benchmark dir or path not found: {benchmark_root}")
    try:
        module = importlib.import_module(f".eid_{eid}", f"{impl_package}")
        solver = getattr(module, module_name)(*args, **kwargs)
    except Exception as e:
        traceback.print_exc()
        print(f"Fail to load implementation", flush=True)
        return
    failed = 0
    for benchmark in tqdm(benchmarks):
        solver.read_benchmark(benchmark)
        try:
            solution = solver.solve()
            if plot:
                solver.plot_solution(
                    solution,
                    filepath=os.path.join(output_root, os.path.basename(benchmark)[:-4] + "_sol.png"),
                )
            if profile:
                profiling = solver.profile(n_runs=5)
            else:
                profiling = 0, 0
            output_path = os.path.join(output_root, os.path.basename(benchmark))
            solver.dump_output_file(*(solution + profiling), output_path)
        except Exception as e:
            traceback.print_exc()
            print(f"Fail to generate output on benchmark: {benchmark}")
            failed += 1
    print(
        f"Finish evaluation for student EID: {eid:10s} Success: {len(benchmarks)-failed} / {len(benchmarks)}"
    )


def score(
    eid: str,
    benchmark_root: str,
    ref_output_root: str,
    output_root: str,
    impl_package: str,
    module_name: str,
    *args,
    **kwargs,
):
    eid = eid.lower()
    output_root = os.path.join(output_root, eid)

    if os.path.isdir(benchmark_root):
        benchmarks = [os.path.join(benchmark_root, i) for i in os.listdir(benchmark_root)]
    elif os.path.isfile(benchmark_root):
        benchmarks = [benchmark_root]
    else:
        raise ValueError(f"Benchmark dir or path not found: {benchmark_root}")

    module = importlib.import_module(f".p2_routing_base", f"{impl_package}")

    solver = getattr(module, module_name)(*args, **kwargs)
    success_list = [["benchmark", "passed", "note", "runtime", "memory"]]
    for benchmark in benchmarks:
        benchmark_name = os.path.basename(benchmark)
        solver.read_benchmark(benchmark)
        output_path = os.path.join(output_root, benchmark_name)
        if not os.path.exists(output_path):
            print(f"Fail to load student {eid} solution {benchmark_name}")
            success = [benchmark_name, False, "SOLUTION_NOT_FOUND", 0, 0]
        else:
            try:
                ref_output_path = os.path.join(ref_output_root, benchmark_name)
                path, wl, wl_list, n_visited_list, runtime, used_mem = solver.load_solution(output_path)
                (
                    ref_path,
                    ref_wl,
                    ref_wl_list,
                    ref_n_visited_list,
                    ref_runtime,
                    ref_used_mem,
                ) = solver.load_solution(ref_output_path)

                if set(solver._split_path(path)) != set(solver._split_path(ref_path)):
                    success = [benchmark_name, False, "PATH_NOT_MATCH", runtime, used_mem]
                else:
                    flag, note = solver.verify_solution(path)
                    if not flag:
                        success = [benchmark_name, False, note, runtime, used_mem]
                    elif wl != ref_wl:
                        success = [benchmark_name, False, "WL_MISMATCH", runtime, used_mem]
                    elif tuple(wl_list) != tuple(ref_wl_list):
                        success = [benchmark_name, False, "WL_LIST_MISMATCH", runtime, used_mem]
                    elif tuple(n_visited_list) != tuple(ref_n_visited_list):
                        success = [benchmark_name, False, "N_VISITED_LIST_MISMATCH", runtime, used_mem]
                    else:
                        max_ratio = 10 # tolerate 10x slower runtime at most
                        ratio = runtime / ref_runtime
                        # smaller than 10x the reference runtime: no penalty
                        # larger than the threshold leads to exponential score decay
                        runtime_weighted_score = np.e ** (1 - max(ratio/max_ratio, 1))
                        success = [benchmark_name, runtime_weighted_score, "PASSED", runtime, used_mem]
            except Exception as e:
                print(f"Fail to score student {eid} solution {benchmark_name}")
                traceback.print_exc()
                success = [benchmark_name, False, "SOLUTION_LOAD_ERROR", 0, 0]
        success_list.append(success)
    passed = sum(i[1] for i in success_list[1:])
    success_list.append(["passed/total", str(passed), str(len(benchmarks)), "-", "-"])
    success_list = np.array(success_list)
    print(f"Finish grading for student EID: {eid:10s} Grade: {passed} / {len(benchmarks)}")
    np.savetxt(os.path.join(output_root, "score.csv"), success_list, fmt="%s", delimiter=",")
    return passed, len(benchmarks)


benchmark_root = "benchmarks"
output_root = "output"
ref_output_root = "output/reference"
impl_package = "student_impl"
module_name = "A_Star_Search"
base_module_name = "A_Star_Search_Base"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--eid", required=True, type=str, help="Your lowercase UT EID, e.g., -e xxxxx")
    parser.add_argument(
        "-b",
        "--benchmark",
        required=False,
        default="all",
        type=str,
        help="One benchmark path or 'all' indicates all benchmarks, e.g., -b ./benchmarks/example_1.txt or -b all. Defaults to 'all'.",
    )
    parser.add_argument(
        "-p",
        "--profile",
        required=False,
        action="store_true",
        default=False,
        help="Whether to perform runtime profiling (Might be slow). Defaults to False. To enable it, add '-p' to your argument",
    )
    parser.add_argument(
        "-r",
        "--run",
        required=False,
        action="store_true",
        help="Whether to execute your implementation. Defaults to False. To enable it, add '-r' to your argument",
    )
    parser.add_argument(
        "-s",
        "--score",
        required=False,
        action="store_true",
        help="Whether to compare with reference solutions and generate scores. Defaults to False. To enable it, add '-s' to your argument",
    )
    parser.add_argument(
        "-d",
        "--draw",
        required=False,
        action="store_true",
        help="Whether to draw the solution for visualization. Defaults to False. To enable it, add '-d' to your argument",
    )

    args = parser.parse_args()
    if args.run:
        evaluate(
            args.eid,
            benchmark_root if args.benchmark.lower() == "all" else args.benchmark,
            output_root,
            impl_package,
            module_name,
            profile=args.profile,
            plot=args.draw,
        )
    if args.score:
        score(
            args.eid,
            benchmark_root if args.benchmark.lower() == "all" else args.benchmark,
            ref_output_root,
            output_root,
            impl_package,
            base_module_name,
        )
