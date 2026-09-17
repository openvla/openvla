"""Regression checks for LIBERO eval episode reseeding (issue #342).

Full sim verification needs LIBERO/robosuite. These tests assert the call-site
invariant without importing the heavy eval stack.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = REPO_ROOT / "experiments" / "robot" / "libero" / "run_libero_eval.py"
UTILS_PATH = REPO_ROOT / "experiments" / "robot" / "libero" / "libero_utils.py"


def _attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _episode_loop_calls(tree: ast.AST) -> list[str]:
    """Return env.* call names inside the num_trials_per_task episode for-loop."""
    calls: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_For(self, node: ast.For) -> None:
            # for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            iter_src = ast.dump(node.iter)
            if "num_trials_per_task" in iter_src:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        name = _attr_name(sub.func)
                        if name is not None and name.startswith("env."):
                            calls.append(name)
                return
            self.generic_visit(node)

    Visitor().visit(tree)
    return calls


class TestLiberoEvalReseed(unittest.TestCase):
    def test_episode_loop_seeds_before_reset(self):
        tree = ast.parse(EVAL_PATH.read_text())
        calls = _episode_loop_calls(tree)
        self.assertIn("env.seed", calls, "episode loop must call env.seed (issue #342)")
        self.assertIn("env.reset", calls)
        self.assertLess(
            calls.index("env.seed"),
            calls.index("env.reset"),
            "env.seed must run before env.reset so fixture body_pos is restored",
        )
        self.assertLess(
            calls.index("env.reset"),
            calls.index("env.set_init_state"),
            "env.reset must run before set_init_state",
        )

    def test_episode_seed_matches_get_libero_env(self):
        """Reseed value must match get_libero_env's env.seed(...) argument."""
        eval_tree = ast.parse(EVAL_PATH.read_text())
        utils_tree = ast.parse(UTILS_PATH.read_text())

        def seed_const_from_calls(tree: ast.AST, *, in_episode_loop: bool) -> int | None:
            found: list[int] = []

            class Visitor(ast.NodeVisitor):
                def visit_For(self, node: ast.For) -> None:
                    if in_episode_loop and "num_trials_per_task" in ast.dump(node.iter):
                        for sub in ast.walk(node):
                            if isinstance(sub, ast.Call) and _attr_name(sub.func) == "env.seed":
                                if sub.args and isinstance(sub.args[0], ast.Constant):
                                    found.append(sub.args[0].value)
                        return
                    self.generic_visit(node)

                def visit_Call(self, node: ast.Call) -> None:
                    if not in_episode_loop and _attr_name(node.func) == "env.seed":
                        if node.args and isinstance(node.args[0], ast.Constant):
                            found.append(node.args[0].value)
                    self.generic_visit(node)

            Visitor().visit(tree)
            return found[0] if found else None

        episode_seed = seed_const_from_calls(eval_tree, in_episode_loop=True)
        utils_seed = seed_const_from_calls(utils_tree, in_episode_loop=False)
        self.assertIsNotNone(episode_seed)
        self.assertIsNotNone(utils_seed)
        self.assertEqual(
            episode_seed,
            utils_seed,
            "episode reseed must match get_libero_env so new-env and reused-env match",
        )


if __name__ == "__main__":
    unittest.main()
