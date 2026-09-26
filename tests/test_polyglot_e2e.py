#!/usr/bin/env python3
"""
End-to-end tests for polyglot repository exploration using tree-sitter.
======================================================================

Tests build_repo_map, build_symbol_index, and rank_relevant_files
on physical non-Python sample repositories (JS/TS and Go).
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.explore import build_repo_map, build_symbol_index, rank_relevant_files, explore


class TestPolyglotE2E(unittest.TestCase):

    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent / "fixtures"

    def test_sample_js_ts_project_end_to_end(self):
        """Verify build_repo_map and tree-sitter build_symbol_index on JS/TS sample repo."""
        repo_path = self.fixtures_dir / "sample_js_project"
        self.assertTrue(repo_path.exists(), f"Path does not exist: {repo_path}")

        # 1. Build repo map
        repo_map = build_repo_map(repo_path)

        # Confirm non-python languages detected
        self.assertIn("typescript", repo_map.detected_languages)
        self.assertIn("javascript", repo_map.detected_languages)
        self.assertNotIn("python", repo_map.detected_languages)

        # Confirm frameworks detected from package.json
        self.assertIn("Express", repo_map.detected_frameworks)
        self.assertIn("Jest", repo_map.detected_frameworks)

        # 2. Build symbol index with tree-sitter
        symbol_index = build_symbol_index(repo_path, repo_map=repo_map)

        # Verify TypeScript symbols extracted via tree-sitter
        ts_rel = "src/middleware/auth.ts"
        self.assertIn(ts_rel, symbol_index.index)
        ts_symbols = {s.name: s for s in symbol_index.index[ts_rel]}

        self.assertIn("AuthMiddleware", ts_symbols)
        self.assertEqual(ts_symbols["AuthMiddleware"].kind, "class")

        self.assertIn("AuthMiddleware.authenticate", ts_symbols)
        self.assertEqual(ts_symbols["AuthMiddleware.authenticate"].kind, "method")
        self.assertTrue(any("authenticate" in (s.signature or "") for s in ts_symbols.values()))

        self.assertIn("AuthMiddleware.validateExpiry", ts_symbols)
        self.assertIn("verifySecret", ts_symbols)
        self.assertEqual(ts_symbols["verifySecret"].kind, "function")

        # Verify JavaScript symbols extracted via tree-sitter
        js_rel = "src/routes/user.js"
        self.assertIn(js_rel, symbol_index.index)
        js_symbols = {s.name: s for s in symbol_index.index[js_rel]}

        self.assertIn("UserController", js_symbols)
        self.assertIn("UserController.getUserById", js_symbols)
        self.assertIn("createRouter", js_symbols)

        # 3. Test rank_relevant_files on JS/TS issue
        issue_text = """
Error: Missing token
    at AuthMiddleware.authenticate (src/middleware/auth.ts:4:13)
    at Layer.handle [as handle_request] (express/lib/router/layer.js:95:5)
"""
        ranked = rank_relevant_files(repo_map, symbol_index, issue_text, top_n=3)
        self.assertGreater(len(ranked), 0)
        self.assertEqual(ranked[0].filepath, "src/middleware/auth.ts")
        self.assertGreater(ranked[0].score, 60.0)

    def test_sample_go_project_end_to_end(self):
        """Verify build_repo_map and tree-sitter build_symbol_index on Go sample repo."""
        repo_path = self.fixtures_dir / "sample_go_project"
        self.assertTrue(repo_path.exists(), f"Path does not exist: {repo_path}")

        # 1. Build repo map
        repo_map = build_repo_map(repo_path)

        # Confirm Go detected
        self.assertIn("go", repo_map.detected_languages)
        self.assertNotIn("python", repo_map.detected_languages)

        # Confirm Gin framework detected from go.mod
        self.assertIn("Gin", repo_map.detected_frameworks)

        # 2. Build symbol index with tree-sitter
        symbol_index = build_symbol_index(repo_path, repo_map=repo_map)

        go_rel = "pkg/orders/service.go"
        self.assertIn(go_rel, symbol_index.index)
        go_symbols = {s.name: s for s in symbol_index.index[go_rel]}

        self.assertIn("Order", go_symbols)
        self.assertIn("OrderService", go_symbols)
        self.assertIn("CreateOrder", go_symbols)
        self.assertIn("CalculateDiscount", go_symbols)

        # Check signature extracted without body
        self.assertTrue(any("CreateOrder" in (s.signature or "") for s in go_symbols.values()))
        self.assertTrue(any("CalculateDiscount" in (s.signature or "") for s in go_symbols.values()))

        # 3. Test ranking on Go issue
        go_issue = "panic: runtime error in CalculateDiscount when rate is negative"
        ranked = rank_relevant_files(repo_map, symbol_index, go_issue, top_n=2)
        self.assertGreater(len(ranked), 0)
        self.assertEqual(ranked[0].filepath, "pkg/orders/service.go")


if __name__ == "__main__":
    unittest.main()
