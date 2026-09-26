#!/usr/bin/env python3
"""
Unit and integration tests for harness.explore
==============================================

Run with::

    .venv/bin/python tests/test_explore.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.explore import (
    build_repo_map,
    build_symbol_index,
    search_code,
    rank_relevant_files,
    explore,
    RepoMap,
    SymbolIndex,
    RankedFile,
    SearchResult,
)


class TestExplore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_repo_map_polyglot_and_configs(self):
        """Test build_repo_map detects frameworks, languages, and builds depth-limited tree."""
        # 1. Setup simulated multi-language repo
        (self.root / "package.json").write_text(
            '{"name": "test-app", "dependencies": {"react": "^18.0.0", "next": "^14.0.0"}, "devDependencies": {"typescript": "^5.0.0", "vitest": "^1.0.0"}}'
        )
        (self.root / "go.mod").write_text("module example.com/auth\n\ngo 1.22\n\nrequire github.com/gin-gonic/gin v1.9.1\n")
        (self.root / "requirements.txt").write_text("fastapi==0.110.0\npytest>=8.0\npydantic>=2.0\n")

        src_dir = self.root / "src"
        src_dir.mkdir()
        (src_dir / "index.ts").write_text("export function main() { console.log('hello'); }\n")
        (src_dir / "server.go").write_text("package main\nfunc main() {}\n")

        # Hidden / ignored directory
        ignored_dir = self.root / "node_modules" / "some_pkg"
        ignored_dir.mkdir(parents=True)
        (ignored_dir / "index.js").write_text("module.exports = {};")

        repo_map = build_repo_map(self.root, max_depth=3)

        # Assertions on RepoMap
        self.assertIn("typescript", repo_map.detected_languages)
        self.assertIn("go", repo_map.detected_languages)
        self.assertIn("python", repo_map.detected_languages)

        # Frameworks detected
        self.assertIn("React", repo_map.detected_frameworks)
        self.assertIn("Next.js", repo_map.detected_frameworks)
        self.assertIn("Vitest", repo_map.detected_frameworks)
        self.assertIn("Gin", repo_map.detected_frameworks)
        self.assertIn("FastAPI", repo_map.detected_frameworks)
        self.assertIn("Pytest", repo_map.detected_frameworks)

        # Tree string
        self.assertIn("src/", repo_map.tree)
        self.assertIn("index.ts", repo_map.tree)
        self.assertNotIn("node_modules", repo_map.tree)

        # File tracking
        tracked_paths = [fi.path for fi in repo_map.files]
        self.assertIn("src/index.ts", tracked_paths)
        self.assertIn("package.json", tracked_paths)
        self.assertNotIn("node_modules/some_pkg/index.js", tracked_paths)

    def test_build_symbol_index_python_and_treesitter(self):
        """Test symbol index extraction for Python (ast) and TS/Go/Rust (tree-sitter)."""
        # Python file
        py_file = self.root / "calculator.py"
        py_file.write_text(
            "class Calculator:\n"
            "    def add(self, a: int, b: int) -> int:\n"
            "        return a + b\n"
            "\n"
            "def global_helper(x: str) -> None:\n"
            "    pass\n"
        )

        # TypeScript file
        ts_file = self.root / "auth.ts"
        ts_file.write_text(
            "export class AuthService {\n"
            "  login(token: string): boolean {\n"
            "    return token.length > 0;\n"
            "  }\n"
            "}\n"
            "export function verifyHash(hash: string): boolean {\n"
            "  return true;\n"
            "}\n"
        )

        # Go file
        go_file = self.root / "service.go"
        go_file.write_text(
            "package service\n\n"
            "type SessionManager struct {\n"
            "  id string\n"
            "}\n\n"
            "func (s *SessionManager) ValidateSession(id string) error {\n"
            "  return nil\n"
            "}\n"
        )

        symbol_index = build_symbol_index(self.root)

        # Check Python symbols
        self.assertIn("calculator.py", symbol_index.index)
        py_syms = {s.name: s for s in symbol_index.index["calculator.py"]}
        self.assertIn("Calculator", py_syms)
        self.assertEqual(py_syms["Calculator"].kind, "class")
        self.assertIn("Calculator.add", py_syms)
        self.assertEqual(py_syms["Calculator.add"].kind, "method")
        self.assertIn("def add(self, a: int, b: int) -> int:", py_syms["Calculator.add"].signature)
        self.assertIn("global_helper", py_syms)

        # Check TypeScript symbols via tree-sitter
        self.assertIn("auth.ts", symbol_index.index)
        ts_syms = {s.name: s for s in symbol_index.index["auth.ts"]}
        self.assertIn("AuthService", ts_syms)
        self.assertIn("AuthService.login", ts_syms)
        self.assertIn("verifyHash", ts_syms)
        self.assertTrue(any("verifyHash" in s.signature for s in symbol_index.index["auth.ts"] if s.signature))

        # Check Go symbols via tree-sitter
        self.assertIn("service.go", symbol_index.index)
        go_syms = {s.name: s for s in symbol_index.index["service.go"]}
        self.assertIn("SessionManager", go_syms)
        self.assertIn("ValidateSession", go_syms)

    def test_search_code_ripgrep(self):
        """Test search_code using ripgrep wrapper with globs and case sensitivity."""
        (self.root / "alpha.txt").write_text("TargetMarker alpha line\nanother line\n")
        (self.root / "beta.py").write_text("# TargetMarker in python file\n")
        (self.root / "gamma.js").write_text("console.log('different');\n")

        # Basic search
        hits = search_code("TargetMarker", path=self.root)
        hit_files = [h.file for h in hits]
        self.assertEqual(len(hits), 2)
        self.assertTrue(any("alpha.txt" in f for f in hit_files))
        self.assertTrue(any("beta.py" in f for f in hit_files))

        # Glob filter
        py_hits = search_code("TargetMarker", path=self.root, glob="*.py")
        self.assertEqual(len(py_hits), 1)
        self.assertTrue(py_hits[0].file.endswith("beta.py"))

        # Case-insensitive
        ci_hits = search_code("targetmarker", path=self.root, case_sensitive=False)
        self.assertEqual(len(ci_hits), 2)

    def test_rank_relevant_files_stack_trace_and_symbols(self):
        """Test rank_relevant_files with stack traces, errors, and symbol matches."""
        # Create dummy project files
        (self.root / "app").mkdir()
        (self.root / "app" / "auth.py").write_text(
            "class AuthenticationHandler:\n"
            "    def authenticate_jwt(self, token: str):\n"
            "        raise TokenExpiredError('expired')\n"
        )
        (self.root / "app" / "db.py").write_text("class Database:\n    pass\n")
        (self.root / "app" / "utils.py").write_text("def sanitize(s): return s\n")

        repo_map = build_repo_map(self.root)
        symbol_index = build_symbol_index(self.root, repo_map=repo_map)

        issue_text = """
Traceback (most recent call last):
  File "app/auth.py", line 3, in authenticate_jwt
    raise TokenExpiredError('expired')
TokenExpiredError: JWT signature has expired
"""
        ranked = rank_relevant_files(repo_map, symbol_index, issue_text, top_n=3)

        self.assertGreater(len(ranked), 0)
        top_file = ranked[0]
        self.assertEqual(top_file.filepath, "app/auth.py")
        self.assertGreater(top_file.score, 50.0)
        self.assertIn("app/auth.py", top_file.reason)

        # Test polymorphic signature
        ranked_alt = rank_relevant_files(issue_text, repo_map, symbol_index, top_n=2)
        self.assertEqual(len(ranked_alt), min(2, len(ranked)))
        self.assertEqual(ranked_alt[0].filepath, "app/auth.py")

    def test_explore_end_to_end(self):
        """Test end-to-end explore wrapper on the current repository."""
        res = explore(self.root, task="fix bug in calculator")
        self.assertIsNotNone(res.repo_map)
        self.assertIsNotNone(res.symbol_index)
        self.assertIsInstance(res.ranked_files, list)


if __name__ == "__main__":
    unittest.main()
