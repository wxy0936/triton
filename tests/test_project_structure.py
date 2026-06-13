import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "common/__init__.py",
    "common/benchmark.py",
    "common/check.py",
    "common/utils.py",
    "chapter_00_setup/00_setup.ipynb",
    "chapter_00_setup/setup_check.py",
    "chapter_01_vector_add/01_vector_add.ipynb",
    "chapter_01_vector_add/vector_add.py",
    "chapter_02_triton_programming_model/02_programming_model.ipynb",
    "chapter_02_triton_programming_model/programming_model.py",
    "chapter_03_mask_and_memory/03_mask_and_memory.ipynb",
    "chapter_03_mask_and_memory/mask_and_memory.py",
    "chapter_04_unary_binary_ops/04_unary_binary_ops.ipynb",
    "chapter_04_unary_binary_ops/unary_binary_ops.py",
    "chapter_05_reduction_sum_max/05_reduction.ipynb",
    "chapter_05_reduction_sum_max/reduction.py",
    "chapter_06_fused_softmax/06_fused_softmax.ipynb",
    "chapter_06_fused_softmax/fused_softmax.py",
    "chapter_07_matmul_baseline/07_matmul_baseline.ipynb",
    "chapter_07_matmul_baseline/matmul_baseline.py",
    "chapter_08_matmul_optimized/08_matmul_optimized.ipynb",
    "chapter_08_matmul_optimized/matmul_optimized.py",
    "chapter_09_autotune_and_benchmark/09_autotune_benchmark.ipynb",
    "chapter_09_autotune_and_benchmark/autotune_benchmark.py",
    "chapter_10_layernorm/10_layernorm.ipynb",
    "chapter_10_layernorm/layernorm.py",
    "chapter_11_attention_intro/11_attention_intro.ipynb",
    "chapter_11_attention_intro/attention_intro.py",
    "chapter_12_online_softmax/12_online_softmax.ipynb",
    "chapter_12_online_softmax/online_softmax.py",
    "chapter_13_flashattention_v1/13_flashattention_v1.ipynb",
    "chapter_13_flashattention_v1/flashattention_v1.py",
    "chapter_14_mini_project/14_mini_project.ipynb",
    "chapter_14_mini_project/mini_project.py",
    "chapter_15_performance_mental_model/15_performance_mental_model.ipynb",
    "chapter_15_performance_mental_model/performance_mental_model.py",
    "chapter_16_profiling_and_benchmarking/16_profiling_and_benchmarking.ipynb",
    "chapter_16_profiling_and_benchmarking/profiling_and_benchmarking.py",
    "chapter_17_attention_memory_analysis/17_attention_memory_analysis.ipynb",
    "chapter_17_attention_memory_analysis/attention_memory_analysis.py",
]

REQUIRED_SYMBOLS = {
    "common/benchmark.py": ["bench"],
    "common/check.py": ["assert_close"],
    "common/utils.py": ["get_device", "print_gpu_info", "set_seed"],
    "chapter_01_vector_add/vector_add.py": ["vector_add_kernel", "vector_add"],
    "chapter_02_triton_programming_model/programming_model.py": [
        "copy_kernel",
        "copy",
        "demo_1d_grid",
        "demo_2d_grid",
    ],
    "chapter_03_mask_and_memory/mask_and_memory.py": [
        "add_contiguous_kernel",
        "add_strided_kernel",
        "add_contiguous",
        "add_strided",
    ],
    "chapter_04_unary_binary_ops/unary_binary_ops.py": [
        "square_kernel",
        "relu_kernel",
        "add_relu_kernel",
        "square",
        "relu",
        "add_relu",
    ],
    "chapter_05_reduction_sum_max/reduction.py": [
        "row_sum_kernel",
        "row_max_kernel",
        "row_sum",
        "row_max",
    ],
    "chapter_06_fused_softmax/fused_softmax.py": ["softmax_kernel", "softmax"],
    "chapter_07_matmul_baseline/matmul_baseline.py": ["matmul_kernel", "matmul"],
    "chapter_08_matmul_optimized/matmul_optimized.py": [
        "matmul_baseline_kernel",
        "matmul_grouped_kernel",
        "matmul_optimized",
    ],
    "chapter_09_autotune_and_benchmark/autotune_benchmark.py": [
        "autotuned_matmul_kernel",
        "autotuned_matmul",
        "benchmark_matmul_shapes",
    ],
    "chapter_10_layernorm/layernorm.py": ["layernorm_kernel", "layernorm"],
    "chapter_11_attention_intro/attention_intro.py": [
        "torch_attention_reference",
        "maybe_compare_with_torch_sdpa",
    ],
    "chapter_12_online_softmax/online_softmax.py": [
        "torch_attention_reference",
        "torch_online_attention",
    ],
    "chapter_13_flashattention_v1/flashattention_v1.py": [
        "torch_attention_reference",
        "flash_attention_kernel",
        "_flash_attention_forward",
        "flash_attention",
    ],
    "chapter_14_mini_project/mini_project.py": [
        "flash_attention_kernel",
        "_flash_attention_forward",
        "tiny_self_attention_torch",
        "tiny_self_attention_triton",
    ],
    "chapter_15_performance_mental_model/performance_mental_model.py": [
        "get_dtype_bytes",
        "estimate_vector_add",
        "estimate_matmul",
        "estimate_softmax",
        "estimate_layernorm",
        "format_bytes",
        "format_flops",
        "benchmark",
        "run_estimation_table",
        "run_actual_benchmarks",
        "main",
    ],
    "chapter_16_profiling_and_benchmarking/profiling_and_benchmarking.py": [
        "require_cuda_or_skip",
        "wrong_timer_example",
        "benchmark_sync",
        "benchmark_cuda_event",
        "benchmark_triton_do_bench",
        "compare_benchmark_methods",
        "TinyMLP",
        "benchmark_forward_backward_train_step",
        "run_torch_profiler_demo",
        "main",
    ],
    "chapter_17_attention_memory_analysis/attention_memory_analysis.py": [
        "format_bytes",
        "estimate_attention_tensors",
        "estimate_flashattention_working_set",
        "print_attention_memory_table",
        "naive_attention_materialized",
        "measure_peak_memory",
        "run_peak_memory_demo",
        "main",
    ],
}


class ProjectStructureTests(unittest.TestCase):
    def test_required_files_exist(self):
        missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
        self.assertEqual(missing, [], f"Missing files: {missing}")

    def test_notebooks_are_real_and_teaching_oriented(self):
        notebooks = [path for path in REQUIRED_FILES if path.endswith(".ipynb")]
        for relative_path in notebooks:
            with self.subTest(notebook=relative_path):
                notebook = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                cell_types = {cell["cell_type"] for cell in notebook["cells"]}
                self.assertIn("markdown", cell_types)
                self.assertIn("code", cell_types)
                self.assertGreaterEqual(len(notebook["cells"]), 6)
                code = "\n".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                )
                if "chapter_16_profiling_and_benchmarking" not in relative_path:
                    self.assertNotIn("time.time(", code)

    def test_python_files_parse_and_chapters_have_main_guards(self):
        scripts = [path for path in REQUIRED_FILES if path.endswith(".py")]
        for relative_path in scripts:
            with self.subTest(script=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                ast.parse(source)
                if "chapter_16_profiling_and_benchmarking" not in relative_path:
                    self.assertNotIn("time.time(", source)
                if relative_path.startswith("chapter_"):
                    self.assertIn('if __name__ == "__main__":', source)

    def test_required_symbols_exist(self):
        for relative_path, expected_names in REQUIRED_SYMBOLS.items():
            with self.subTest(script=relative_path):
                tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
                names = {
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                }
                self.assertEqual(set(expected_names) - names, set())

    def test_requirements_include_core_packages(self):
        requirements = {
            line.strip().split("=")[0].split(">")[0].split("<")[0]
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        self.assertTrue({"torch", "triton", "jupyter", "notebook"} <= requirements)

    def test_no_later_chapter_code_exists(self):
        later = [
            path.name
            for path in ROOT.glob("chapter_*")
            if path.is_dir() and int(path.name.split("_")[1]) >= 18
        ]
        self.assertEqual(later, [])

    def test_setup_check_reports_missing_runtime_clearly(self):
        result = subprocess.run(
            [sys.executable, "chapter_00_setup/setup_check.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertNotIn("ModuleNotFoundError", output)
        self.assertTrue(
            any(
                message in output
                for message in (
                    "Triton tutorial environment is ready",
                    "CUDA is not available",
                    "Triton is not installed",
                )
            ),
            output,
        )

    def test_programming_model_uses_portable_power_of_two_check(self):
        source = (ROOT / "chapter_02_triton_programming_model/programming_model.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("triton.is_power_of_2", source)

    def test_benchmark_explicitly_returns_median(self):
        source = (ROOT / "common/benchmark.py").read_text(encoding="utf-8")
        self.assertIn('return_mode="median"', source)

    def test_flashattention_core_copies_do_not_drift(self):
        def normalized_function(relative_path, function_name):
            tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            function.decorator_list = []
            return ast.dump(function, include_attributes=False)

        chapter_13 = "chapter_13_flashattention_v1/flashattention_v1.py"
        chapter_14 = "chapter_14_mini_project/mini_project.py"
        for function_name in ("flash_attention_kernel", "_flash_attention_forward"):
            with self.subTest(function=function_name):
                self.assertEqual(
                    normalized_function(chapter_13, function_name),
                    normalized_function(chapter_14, function_name),
                )

    def test_flashattention_chapters_are_independent(self):
        for relative_path in (
            "chapter_13_flashattention_v1/flashattention_v1.py",
            "chapter_14_mini_project/mini_project.py",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("common.attention", source)
            self.assertNotIn("chapter_13", source)
            self.assertNotIn("chapter_14", source)

    def test_flashattention_correctness_compares_float_values(self):
        for relative_path in (
            "chapter_13_flashattention_v1/flashattention_v1.py",
            "chapter_14_mini_project/mini_project.py",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("actual.float()", source)
            self.assertIn("expected.float()", source)

    def test_online_attention_guards_zero_denominator(self):
        source = (ROOT / "chapter_12_online_softmax/online_softmax.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("torch.where(l > 0", source)

    def test_wrong_timer_is_the_only_unsynchronized_time_time_use(self):
        path = ROOT / "chapter_16_profiling_and_benchmarking/profiling_and_benchmarking.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = []
        for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
            uses_time_time = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "time"
                and node.func.attr == "time"
                for node in ast.walk(function)
            )
            if uses_time_time:
                offenders.append(function.name)
        self.assertEqual(offenders, ["wrong_timer_example"])

    def test_formal_benchmark_helpers_use_correct_cuda_timing(self):
        path = ROOT / "chapter_16_profiling_and_benchmarking/profiling_and_benchmarking.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("torch.cuda.synchronize", functions["benchmark_sync"])
        self.assertIn("torch.cuda.Event", functions["benchmark_cuda_event"])
        self.assertIn("triton.testing.do_bench", functions["benchmark_triton_do_bench"])


if __name__ == "__main__":
    unittest.main()
