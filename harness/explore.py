"""
harness.explore
===============

Phase 1 — Explore: Repository mapping, AST symbol indexing, and relevance ranking.

Before the agent can plan changes it needs a mental model of the codebase.
This module builds three complementary views:

1. **Repo map** — a lightweight tree of every file with metadata (size,
   language, last-modified) and auto-detection of languages & frameworks.
2. **Symbol index** — an AST-derived index of classes, functions, and methods
   (Python ``ast`` for .py files, ``tree-sitter`` for polyglot languages)
   without loading full file bodies into memory.
3. **Code search** — ripgrep wrapper for high-performance pattern search.
4. **Relevance ranking** — given the task/issue description, scores and ranks
   files using stack trace paths, error messages, symbol matches, and search results.

Typical usage::

    repo_map = build_repo_map(Path("."))
    symbols  = build_symbol_index(Path("."))
    ranked   = rank_relevant_files(repo_map, symbols, "Fix auth bug")
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fnmatch
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    """Metadata for a single file in the repository.

    Attributes:
        path:          Relative path from the repo root.
        language:      Detected programming language (e.g. ``"python"``).
        size_bytes:    File size in bytes.
        last_modified: ISO-8601 timestamp of last modification.
    """

    path: str
    language: str | None = None
    size_bytes: int = 0
    last_modified: str | None = None


@dataclass
class RepoMap:
    """Structural map of a repository.

    Attributes:
        root:                 Absolute path to the repository root.
        files:                Flat list of every tracked file and its metadata.
        tree:                 Depth-limited formatted directory tree.
        detected_languages:   List of detected programming languages.
        detected_frameworks: List of detected frameworks/libraries.
    """

    root: Path
    files: list[FileInfo] = field(default_factory=list)
    tree: str = ""
    detected_languages: list[str] = field(default_factory=list)
    detected_frameworks: list[str] = field(default_factory=list)


@dataclass
class Symbol:
    """A single code symbol extracted via AST parsing.

    Attributes:
        name:       Qualified name (e.g. ``"MyClass.my_method"``).
        kind:       Symbol kind — ``"class"``, ``"function"``, ``"method"``, etc.
        line_start: First line of the definition (1-indexed).
        line_end:   Last line of the definition (1-indexed).
        signature:  Extracted declaration signature without body.
    """

    name: str
    kind: str
    line_start: int
    line_end: int
    signature: str | None = None


@dataclass
class SymbolIndex:
    """Mapping from file paths to their extracted symbols.

    Attributes:
        index: ``{relative_filepath: [Symbol, …]}``.
    """

    index: dict[str, list[Symbol]] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single match from code search (e.g. ripgrep).

    Attributes:
        file:    Relative path from the repo root.
        line:    1-indexed line number.
        content: Line content of the match.
    """

    file: str
    line: int
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "line": self.line, "content": self.content}


@dataclass
class RankedFile:
    """A file scored for relevance to the current task.

    Attributes:
        filepath: Relative path from the repo root.
        score:    Relevance score (higher = more relevant).
        reason:   Short human-readable justification for the score.
    """

    filepath: str
    score: float
    reason: str


@dataclass
class ExploreResult:
    """Aggregated output of the Explore phase.

    Passed downstream to :func:`harness.plan.generate_plan` as context.

    Attributes:
        repo_map:      The full repository map.
        symbol_index:  AST-derived symbol index.
        ranked_files:  Files ranked by relevance to the task.
    """

    repo_map: RepoMap
    symbol_index: SymbolIndex
    ranked_files: list[RankedFile]


# ---------------------------------------------------------------------------
# Constants & Language mappings
# ---------------------------------------------------------------------------

DEFAULT_IGNORE_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "out",
    "target",
    "vendor",
    ".idea",
    ".vscode",
    ".tox",
    ".next",
    ".nuxt",
    ".turbo",
    "bin",
    "obj",
    "coverage",
    ".coverage",
    ".cache",
}

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown",
    ".rst": "rst",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".dart": "dart",
    ".lua": "lua",
}

MAX_PARSE_FILE_SIZE_BYTES = 512 * 1024  # 512 KB limit for symbol extraction


# ---------------------------------------------------------------------------
# Gitignore parser helper
# ---------------------------------------------------------------------------

def _load_gitignore_patterns(repo_root: Path) -> list[str]:
    """Read .gitignore at the repository root and return valid pattern lines."""
    gitignore_path = repo_root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    try:
        lines = gitignore_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    except Exception:
        pass
    return patterns


def _is_ignored(rel_path: str, is_dir: bool, gitignore_patterns: list[str]) -> bool:
    """Check if a path matches standard ignore directories or .gitignore patterns."""
    parts = rel_path.split("/")
    # Check ignore directories in any segment of the path
    for part in parts:
        if part in DEFAULT_IGNORE_DIRS:
            return True

    # Check gitignore patterns
    for pat in gitignore_patterns:
        pat_clean = pat.rstrip("/")
        # Path segment exact match or fnmatch
        if fnmatch.fnmatch(rel_path, pat_clean) or fnmatch.fnmatch(parts[-1], pat_clean):
            return True
        if fnmatch.fnmatch(rel_path, f"*{pat_clean}*"):
            return True
        if is_dir and fnmatch.fnmatch(f"{rel_path}/", pat):
            return True

    return False


# ---------------------------------------------------------------------------
# Language & Framework detection
# ---------------------------------------------------------------------------

def _detect_file_language(file_path: Path) -> str | None:
    """Detect language from file extension or shebang."""
    ext = file_path.suffix.lower()
    if ext in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[ext]

    # Special filenames
    name = file_path.name.lower()
    if name in ("dockerfile", "containerfile"):
        return "dockerfile"
    if name in ("makefile", "gnumakefile"):
        return "make"

    # Check shebang for extensionless files
    try:
        with open(file_path, "rb") as f:
            first_line = f.readline(128).decode("utf-8", errors="ignore").strip()
            if first_line.startswith("#!"):
                if "python" in first_line:
                    return "python"
                if "node" in first_line:
                    return "javascript"
                if "bash" in first_line or "sh" in first_line:
                    return "shell"
                if "ruby" in first_line:
                    return "ruby"
                if "perl" in first_line:
                    return "perl"
    except Exception:
        pass

    return None


def detect_languages_and_frameworks(
    repo_root: Path, file_infos: list[FileInfo]
) -> tuple[list[str], list[str]]:
    """Detect repo-level languages and frameworks from configuration files.

    Inspects package.json, requirements.txt, pyproject.toml, go.mod, Cargo.toml,
    pom.xml, build.gradle, Gemfile, composer.json, etc.
    """
    detected_languages: set[str] = set()
    detected_frameworks: set[str] = set()

    # 1. package.json (Node / JS / TS)
    pkg_json_path = repo_root / "package.json"
    if pkg_json_path.is_file():
        try:
            data = json.loads(pkg_json_path.read_text(encoding="utf-8", errors="replace"))
            all_deps = {
                **data.get("dependencies", {}),
                **data.get("devDependencies", {}),
                **data.get("peerDependencies", {}),
            }

            if "typescript" in all_deps or (repo_root / "tsconfig.json").is_file():
                detected_languages.add("typescript")
            else:
                detected_languages.add("javascript")

            framework_map = {
                "react": "React",
                "next": "Next.js",
                "vue": "Vue",
                "nuxt": "Nuxt",
                "svelte": "Svelte",
                "@angular/core": "Angular",
                "express": "Express",
                "fastify": "Fastify",
                "@nestjs/core": "NestJS",
                "vite": "Vite",
                "webpack": "Webpack",
                "jest": "Jest",
                "vitest": "Vitest",
                "mocha": "Mocha",
                "cypress": "Cypress",
                "playwright": "Playwright",
                "tailwindcss": "TailwindCSS",
                "electron": "Electron",
                "prisma": "Prisma",
            }
            for dep_key, fw_name in framework_map.items():
                if dep_key in all_deps:
                    detected_frameworks.add(fw_name)
        except Exception:
            detected_languages.add("javascript")

    # 2. Python (requirements.txt, pyproject.toml, setup.py, Pipfile)
    py_configs = ["requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile"]
    for py_cfg in py_configs:
        p = repo_root / py_cfg
        if p.is_file():
            detected_languages.add("python")
            try:
                content = p.read_text(encoding="utf-8", errors="replace").lower()
                py_fw_map = {
                    "django": "Django",
                    "flask": "Flask",
                    "fastapi": "FastAPI",
                    "pytest": "Pytest",
                    "torch": "PyTorch",
                    "pytorch": "PyTorch",
                    "tensorflow": "TensorFlow",
                    "pandas": "Pandas",
                    "numpy": "NumPy",
                    "sqlalchemy": "SQLAlchemy",
                    "celery": "Celery",
                    "pydantic": "Pydantic",
                    "typer": "Typer",
                    "click": "Click",
                    "scikit-learn": "Scikit-Learn",
                }
                for key, fw_name in py_fw_map.items():
                    if re.search(rf"\b{re.escape(key)}\b", content):
                        detected_frameworks.add(fw_name)
            except Exception:
                pass

    # 3. go.mod (Go)
    go_mod_path = repo_root / "go.mod"
    if go_mod_path.is_file():
        detected_languages.add("go")
        try:
            content = go_mod_path.read_text(encoding="utf-8", errors="replace")
            go_fw_map = {
                "github.com/gin-gonic/gin": "Gin",
                "github.com/labstack/echo": "Echo",
                "github.com/gofiber/fiber": "Fiber",
                "github.com/go-chi/chi": "Chi",
                "github.com/spf13/cobra": "Cobra",
                "gorm.io/gorm": "GORM",
                "google.golang.org/grpc": "gRPC",
                "github.com/gorilla/mux": "Gorilla Mux",
            }
            for mod_url, fw_name in go_fw_map.items():
                if mod_url in content:
                    detected_frameworks.add(fw_name)
        except Exception:
            pass

    # 4. Cargo.toml (Rust)
    cargo_path = repo_root / "Cargo.toml"
    if cargo_path.is_file():
        detected_languages.add("rust")
        try:
            content = cargo_path.read_text(encoding="utf-8", errors="replace").lower()
            rust_fw_map = {
                "actix-web": "Actix Web",
                "axum": "Axum",
                "rocket": "Rocket",
                "tokio": "Tokio",
                "serde": "Serde",
                "diesel": "Diesel",
                "sqlx": "SQLx",
                "clap": "Clap",
            }
            for crate, fw_name in rust_fw_map.items():
                if crate in content:
                    detected_frameworks.add(fw_name)
        except Exception:
            pass

    # 5. Java / Kotlin (pom.xml, build.gradle)
    if (repo_root / "pom.xml").is_file() or (repo_root / "build.gradle").is_file() or (repo_root / "build.gradle.kts").is_file():
        detected_languages.add("java")
        gradle_kts = repo_root / "build.gradle.kts"
        if gradle_kts.is_file():
            detected_languages.add("kotlin")
        for f in (repo_root / "pom.xml", repo_root / "build.gradle", gradle_kts):
            if f.is_file():
                try:
                    c = f.read_text(encoding="utf-8", errors="replace").lower()
                    if "spring-boot" in c or "org.springframework" in c:
                        detected_frameworks.add("Spring Boot")
                    if "quarkus" in c:
                        detected_frameworks.add("Quarkus")
                    if "micronaut" in c:
                        detected_frameworks.add("Micronaut")
                    if "junit" in c:
                        detected_frameworks.add("JUnit")
                except Exception:
                    pass

    # 6. Ruby (Gemfile)
    gemfile_path = repo_root / "Gemfile"
    if gemfile_path.is_file():
        detected_languages.add("ruby")
        try:
            c = gemfile_path.read_text(encoding="utf-8", errors="replace").lower()
            if "rails" in c:
                detected_frameworks.add("Rails")
            if "sinatra" in c:
                detected_frameworks.add("Sinatra")
            if "rspec" in c:
                detected_frameworks.add("RSpec")
        except Exception:
            pass

    # 7. PHP (composer.json)
    composer_path = repo_root / "composer.json"
    if composer_path.is_file():
        detected_languages.add("php")
        try:
            c = composer_path.read_text(encoding="utf-8", errors="replace").lower()
            if "laravel" in c:
                detected_frameworks.add("Laravel")
            if "symfony" in c:
                detected_frameworks.add("Symfony")
            if "phpunit" in c:
                detected_frameworks.add("PHPUnit")
        except Exception:
            pass

    # 8. C / C++ (CMakeLists.txt, Makefile)
    if (repo_root / "CMakeLists.txt").is_file() or (repo_root / "Makefile").is_file():
        detected_languages.add("c/c++")

    # 9. Tally languages from actual files if config files didn't capture them
    lang_counts: dict[str, int] = {}
    for fi in file_infos:
        if fi.language:
            lang_counts[fi.language] = lang_counts.get(fi.language, 0) + 1

    sorted_langs = sorted(lang_counts.keys(), key=lambda l: lang_counts[l], reverse=True)
    for l in sorted_langs:
        detected_languages.add(l)

    return sorted(detected_languages), sorted(detected_frameworks)


# ---------------------------------------------------------------------------
# Directory tree builder
# ---------------------------------------------------------------------------

def _build_directory_tree(
    repo_root: Path,
    gitignore_patterns: list[str],
    max_depth: int = 4,
    max_files_per_dir: int = 30,
) -> str:
    """Build a depth-limited formatted ASCII directory tree string."""
    lines: list[str] = [f"{repo_root.name}/"]

    def _walk(dir_path: Path, prefix: str, current_depth: int) -> None:
        if current_depth > max_depth:
            return

        try:
            entries = sorted(list(dir_path.iterdir()), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        # Filter ignored
        filtered_entries: list[Path] = []
        for entry in entries:
            name = entry.name
            if name.startswith(".") and name not in (".gitignore", ".env.example"):
                continue
            rel = str(entry.relative_to(repo_root))
            if _is_ignored(rel, entry.is_dir(), gitignore_patterns):
                continue
            filtered_entries.append(entry)

        total_entries = len(filtered_entries)
        displayed_entries = filtered_entries[:max_files_per_dir]
        has_more = total_entries > max_files_per_dir

        for idx, entry in enumerate(displayed_entries):
            is_last = (idx == len(displayed_entries) - 1) and not has_more
            connector = "└── " if is_last else "├── "
            entry_display = f"{entry.name}/" if entry.is_dir() else entry.name
            lines.append(f"{prefix}{connector}{entry_display}")

            if entry.is_dir() and current_depth < max_depth:
                new_prefix = prefix + ("    " if is_last else "│   ")
                _walk(entry, new_prefix, current_depth + 1)

        if has_more:
            remaining = total_entries - max_files_per_dir
            lines.append(f"{prefix}└── ... [{remaining} more entries]")

    _walk(repo_root, "", 1)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tree-sitter polyglot symbol extractor
# ---------------------------------------------------------------------------

class TreeSitterHelper:
    """Polyglot symbol extractor powered by tree-sitter.

    Extracts function, method, class, struct, and interface signatures
    WITHOUT keeping full file bodies in memory.
    """

    _parsers: dict[str, Any] = {}
    _init_attempted: set[str] = set()

    @classmethod
    def get_parser(cls, language: str) -> Any | None:
        """Lazily initialize and return a tree-sitter Parser for the given language."""
        lang_key = language.lower()
        if lang_key in cls._parsers:
            return cls._parsers[lang_key]

        if lang_key in cls._init_attempted:
            return None

        cls._init_attempted.add(lang_key)

        try:
            from tree_sitter import Language, Parser

            if lang_key in ("javascript", "js"):
                import tree_sitter_javascript
                p = Parser(Language(tree_sitter_javascript.language()))
            elif lang_key in ("typescript", "ts"):
                import tree_sitter_typescript
                p = Parser(Language(tree_sitter_typescript.language_typescript()))
            elif lang_key == "tsx":
                import tree_sitter_typescript
                p = Parser(Language(tree_sitter_typescript.language_tsx()))
            elif lang_key == "go":
                import tree_sitter_go
                p = Parser(Language(tree_sitter_go.language()))
            elif lang_key == "rust":
                import tree_sitter_rust
                p = Parser(Language(tree_sitter_rust.language()))
            elif lang_key == "java":
                import tree_sitter_java
                p = Parser(Language(tree_sitter_java.language()))
            elif lang_key in ("c", "h"):
                import tree_sitter_c
                p = Parser(Language(tree_sitter_c.language()))
            elif lang_key in ("cpp", "hpp", "cc", "cxx"):
                import tree_sitter_cpp
                p = Parser(Language(tree_sitter_cpp.language()))
            elif lang_key in ("csharp", "cs"):
                import tree_sitter_c_sharp
                p = Parser(Language(tree_sitter_c_sharp.language()))
            elif lang_key in ("ruby", "rb"):
                import tree_sitter_ruby
                p = Parser(Language(tree_sitter_ruby.language()))
            elif lang_key == "php":
                import tree_sitter_php
                p = Parser(Language(tree_sitter_php.language_php()))
            else:
                return None

            cls._parsers[lang_key] = p
            return p
        except Exception:
            return None

    @classmethod
    def extract_symbols(cls, source_bytes: bytes, language: str) -> list[Symbol]:
        """Extract symbol signatures from source bytes using tree-sitter."""
        parser = cls.get_parser(language)
        if not parser:
            return []

        try:
            tree = parser.parse(source_bytes)
            symbols: list[Symbol] = []
            cls._extract_node_symbols(tree.root_node, source_bytes, language, symbols, parent_scope=None)
            return symbols
        except Exception:
            return []

    @classmethod
    def _extract_node_symbols(
        cls,
        node: Any,
        source: bytes,
        language: str,
        symbols: list[Symbol],
        parent_scope: str | None = None,
    ) -> None:
        """Traverse AST nodes and extract classes, functions, and methods."""
        node_type = node.type

        # 1. Classes / Structs / Interfaces / Traits
        is_class = node_type in (
            "class_declaration",
            "class_specifier",
            "struct_item",
            "enum_item",
            "trait_item",
            "interface_declaration",
            "type_declaration",
            "class",
            "module",
        )

        # 2. Functions / Methods
        is_func = node_type in (
            "function_declaration",
            "function_definition",
            "method_declaration",
            "method_definition",
            "function_item",
            "singleton_method",
        )

        current_scope = parent_scope

        if is_class:
            name_node = node.child_by_field_name("name")
            name_str = (
                source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                if name_node
                else None
            )

            # In Go, type_declaration contains type_spec
            if not name_str and node_type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        n = child.child_by_field_name("name")
                        if n:
                            name_str = source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
                            break

            if name_str:
                qual_name = f"{parent_scope}.{name_str}" if parent_scope else name_str
                body_node = node.child_by_field_name("body")
                if body_node:
                    sig = source[node.start_byte:body_node.start_byte].decode("utf-8", errors="replace").strip()
                else:
                    sig = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace").splitlines()[0].strip()

                symbols.append(
                    Symbol(
                        name=qual_name,
                        kind="class" if "interface" not in node_type else "interface",
                        line_start=node.start_point.row + 1,
                        line_end=node.end_point.row + 1,
                        signature=sig,
                    )
                )
                current_scope = qual_name

        elif is_func:
            name_node = node.child_by_field_name("name")
            name_str = (
                source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                if name_node
                else None
            )

            if name_str:
                qual_name = f"{parent_scope}.{name_str}" if parent_scope else name_str
                body_node = node.child_by_field_name("body")
                if body_node:
                    sig = source[node.start_byte:body_node.start_byte].decode("utf-8", errors="replace").strip()
                else:
                    sig = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace").splitlines()[0].strip()

                kind = "method" if parent_scope else "function"
                symbols.append(
                    Symbol(
                        name=qual_name,
                        kind=kind,
                        line_start=node.start_point.row + 1,
                        line_end=node.end_point.row + 1,
                        signature=sig,
                    )
                )

        elif node_type == "impl_item" and language == "rust":
            # Rust impl blocks
            type_node = node.child_by_field_name("type")
            if type_node:
                current_scope = source[type_node.start_byte:type_node.end_byte].decode("utf-8", errors="replace").strip()

        # Recurse into children
        for child in node.children:
            cls._extract_node_symbols(child, source, language, symbols, parent_scope=current_scope)


# ---------------------------------------------------------------------------
# Python AST symbol extractor
# ---------------------------------------------------------------------------

def _extract_python_symbols(source: str) -> list[Symbol]:
    """Extract Python symbols using the standard library ast module.

    Extracts function and class signatures WITHOUT storing bodies in memory.
    """
    symbols: list[Symbol] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _regex_fallback_extract_symbols(source, "python")

    def _format_args(args_node: ast.arguments) -> str:
        try:
            return ast.unparse(args_node)
        except Exception:
            return "..."

    def _format_return(returns_node: ast.AST | None) -> str:
        if not returns_node:
            return ""
        try:
            return f" -> {ast.unparse(returns_node)}"
        except Exception:
            return ""

    def _format_bases(bases: list[ast.expr]) -> str:
        if not bases:
            return ""
        try:
            return f"({', '.join(ast.unparse(b) for b in bases)})"
        except Exception:
            return "(...)"

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_sig = f"class {node.name}{_format_bases(node.bases)}:"
            symbols.append(
                Symbol(
                    name=node.name,
                    kind="class",
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno),
                    signature=class_sig,
                )
            )

            # Methods inside class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                    method_sig = f"{prefix} {item.name}({_format_args(item.args)}){_format_return(item.returns)}:"
                    symbols.append(
                        Symbol(
                            name=f"{node.name}.{item.name}",
                            kind="method",
                            line_start=item.lineno,
                            line_end=getattr(item, "end_lineno", item.lineno),
                            signature=method_sig,
                        )
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            func_sig = f"{prefix} {node.name}({_format_args(node.args)}){_format_return(node.returns)}:"
            symbols.append(
                Symbol(
                    name=node.name,
                    kind="function",
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno),
                    signature=func_sig,
                )
            )

    return symbols


# ---------------------------------------------------------------------------
# Fallback regex symbol extractor (resilient for any language)
# ---------------------------------------------------------------------------

def _regex_fallback_extract_symbols(source: str, language: str | None = None) -> list[Symbol]:
    """Lightweight regex-based fallback to extract declarations line-by-line."""
    symbols: list[Symbol] = []
    lines = source.splitlines()

    class_pat = re.compile(r"^\s*(?:export\s+|pub\s+)?(?:class|struct|interface|trait|type)\s+([A-Za-z0-9_]+)")
    func_pat = re.compile(r"^\s*(?:async\s+)?(?:export\s+|pub\s+|public\s+|private\s+|protected\s+|static\s+)*(?:def|function|func|fn)\s+([A-Za-z0-9_]+)")

    for idx, line in enumerate(lines, start=1):
        c_m = class_pat.match(line)
        if c_m:
            symbols.append(
                Symbol(
                    name=c_m.group(1),
                    kind="class",
                    line_start=idx,
                    line_end=idx,
                    signature=line.strip(),
                )
            )
            continue

        f_m = func_pat.match(line)
        if f_m:
            symbols.append(
                Symbol(
                    name=f_m.group(1),
                    kind="function",
                    line_start=idx,
                    line_end=idx,
                    signature=line.strip(),
                )
            )

    return symbols


# ---------------------------------------------------------------------------
# Public API: 1. build_repo_map
# ---------------------------------------------------------------------------

def build_repo_map(repo_path: Path | str, max_depth: int = 4) -> RepoMap:
    """Walk the directory tree and produce a structural map of the repository.

    Detects languages, frameworks, and formats a depth-limited directory tree.
    Respects .gitignore rules and common ignore patterns.

    Args:
        repo_path: Path to the repository root.
        max_depth: Maximum recursion depth for the formatted directory tree.

    Returns:
        A :class:`RepoMap` containing file metadata, detected languages/frameworks,
        and tree view.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        return RepoMap(root=root, files=[], tree="", detected_languages=[], detected_frameworks=[])

    gitignore_patterns = _load_gitignore_patterns(root)
    file_infos: list[FileInfo] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        rel_dir = str(current_dir.relative_to(root))
        if rel_dir == ".":
            rel_dir = ""

        # Filter out ignored directories in-place to stop os.walk descending
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d not in DEFAULT_IGNORE_DIRS
            and not _is_ignored(f"{rel_dir}/{d}".lstrip("/"), True, gitignore_patterns)
        ]

        for fname in filenames:
            rel_file = f"{rel_dir}/{fname}".lstrip("/")
            if _is_ignored(rel_file, False, gitignore_patterns):
                continue

            full_path = current_dir / fname
            try:
                st = full_path.stat()
                mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
                size = st.st_size
            except Exception:
                mtime = None
                size = 0

            lang = _detect_file_language(full_path)
            file_infos.append(
                FileInfo(
                    path=rel_file,
                    language=lang,
                    size_bytes=size,
                    last_modified=mtime,
                )
            )

    # Sort files deterministically
    file_infos.sort(key=lambda fi: fi.path)

    # Detect languages & frameworks from config files and file tally
    languages, frameworks = detect_languages_and_frameworks(root, file_infos)

    # Generate tree view
    tree_str = _build_directory_tree(root, gitignore_patterns, max_depth=max_depth)

    return RepoMap(
        root=root,
        files=file_infos,
        tree=tree_str,
        detected_languages=languages,
        detected_frameworks=frameworks,
    )


# ---------------------------------------------------------------------------
# Public API: 2. build_symbol_index
# ---------------------------------------------------------------------------

def build_symbol_index(
    repo_path: Path | str,
    repo_map: RepoMap | None = None,
) -> SymbolIndex:
    """Parse source files and extract an AST-based symbol index.

    Uses Python's ``ast`` module for ``.py`` files and ``tree-sitter`` for
    other supported languages (JS, TS, Go, Rust, Java, C/C++, C#, Ruby, PHP)
    to extract function/class signatures WITHOUT storing full file bodies.

    Args:
        repo_path: Absolute or relative path to the repository root.
        repo_map:  Optional pre-built :class:`RepoMap` to avoid re-walking.

    Returns:
        A :class:`SymbolIndex` mapping relative file paths to lists of symbols.
    """
    root = Path(repo_path).resolve()
    if repo_map is None:
        repo_map = build_repo_map(root)

    index: dict[str, list[Symbol]] = {}

    for fi in repo_map.files:
        # Skip files exceeding size limit or without detected programming language
        if fi.size_bytes > MAX_PARSE_FILE_SIZE_BYTES or not fi.language:
            continue

        # Skip minified files
        if fi.path.endswith((".min.js", ".min.css", ".map")):
            continue

        full_path = root / fi.path
        if not full_path.is_file():
            continue

        symbols: list[Symbol] = []
        try:
            if fi.language == "python":
                source_text = full_path.read_text(encoding="utf-8", errors="replace")
                symbols = _extract_python_symbols(source_text)
            else:
                # Tree-sitter for other languages
                source_bytes = full_path.read_bytes()
                symbols = TreeSitterHelper.extract_symbols(source_bytes, fi.language)

                # If tree-sitter yielded nothing or language grammar not installed, regex fallback
                if not symbols:
                    source_text = source_bytes.decode("utf-8", errors="replace")
                    symbols = _regex_fallback_extract_symbols(source_text, fi.language)
        except Exception:
            pass

        if symbols:
            index[fi.path] = symbols

    return SymbolIndex(index=index)


# ---------------------------------------------------------------------------
# Public API: 3. search_code (ripgrep wrapper)
# ---------------------------------------------------------------------------

def _find_ripgrep_executable() -> str | None:
    """Locate ripgrep (rg) binary in PATH or common installation directories."""
    rg_in_path = shutil.which("rg")
    if rg_in_path:
        return rg_in_path

    candidates = [
        "/opt/homebrew/bin/rg",
        "/usr/local/bin/rg",
        "/usr/bin/rg",
        os.path.expanduser("~/.cargo/bin/rg"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def search_code(
    pattern: str,
    path: Path | str | None = None,
    glob: str | None = None,
    max_results: int = 100,
    case_sensitive: bool = False,
) -> list[SearchResult]:
    """Wrap ripgrep via subprocess to execute fast code searches.

    Falls back gracefully to git grep or Python regex search if ripgrep is unavailable.

    Args:
        pattern:        Regular expression or literal string to search for.
        path:           Directory or file path to search within (defaults to current dir).
        glob:           Optional glob pattern to filter files (e.g. ``"*.py"``).
        max_results:    Maximum number of matching lines to return.
        case_sensitive: If False, performs case-insensitive search.

    Returns:
        List of :class:`SearchResult` objects containing file, line, and content.
    """
    search_path = Path(path).resolve() if path else Path.cwd()
    rg_bin = _find_ripgrep_executable()

    if rg_bin:
        cmd = [
            rg_bin,
            "--line-number",
            "--no-heading",
            "--color=never",
            f"--max-count={max_results}",
        ]
        if not case_sensitive:
            cmd.append("-i")
        if glob:
            cmd.extend(["--glob", glob])

        cmd.extend(["--", pattern, str(search_path)])

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=15,
            )
            results: list[SearchResult] = []
            for line in res.stdout.splitlines():
                if not line:
                    continue
                # Output format: <file>:<line>:<content>
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_str, line_num, content = parts[0], parts[1], parts[2]
                    try:
                        line_int = int(line_num)
                        # Normalize file path relative to search_path
                        p_file = Path(file_str)
                        if p_file.is_absolute() and search_path in p_file.parents:
                            rel_file = str(p_file.relative_to(search_path))
                        else:
                            rel_file = file_str
                        results.append(SearchResult(file=rel_file, line=line_int, content=content))
                        if len(results) >= max_results:
                            break
                    except ValueError:
                        continue
            return results
        except Exception:
            pass

    # Fallback: Python regex search across files
    results = []
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error:
        regex = re.compile(re.escape(pattern), flags)

    if search_path.is_file():
        candidates = [search_path]
    else:
        candidates = [
            p for p in search_path.rglob(glob or "*")
            if p.is_file() and not any(part in DEFAULT_IGNORE_DIRS for part in p.parts)
        ]

    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f, start=1):
                    if regex.search(line):
                        rel_file = str(p.relative_to(search_path)) if search_path != p else p.name
                        results.append(SearchResult(file=rel_file, line=line_idx, content=line.rstrip("\r\n")))
                        if len(results) >= max_results:
                            return results
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Public API: 4. rank_relevant_files
# ---------------------------------------------------------------------------

def _extract_issue_clues(issue_text: str) -> dict[str, Any]:
    """Extract paths, error messages, identifiers, and distinctive terms from issue text."""
    # 1. Stack trace / file path patterns
    # e.g., File "src/foo.py", line 42 | at Object.<anonymous> (test/bar.ts:12) | models.py:10
    path_regex = re.compile(r"""(?:File\s+["']|at\s+(?:.*?\()|[\s(\'"`])([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9_]{1,10})(?::(\d+))?""")
    explicit_paths: set[str] = set()
    for m in path_regex.finditer(issue_text):
        candidate = m.group(1).replace("\\", "/").strip("./")
        if "/" in candidate or any(candidate.endswith(ext) for ext in EXTENSION_TO_LANGUAGE):
            explicit_paths.add(candidate)

    # 2. Error and exception names (e.g. ValueError, AuthError, NullPointerException)
    error_regex = re.compile(r"\b([A-Z][a-zA-Z0-9_]*(?:Error|Exception|Failure|Fault))\b")
    error_names = set(error_regex.findall(issue_text))

    # 3. Quoted identifiers & backticked symbols: `my_func`, "MyClass"
    code_span_regex = re.compile(r"`([^`]+)`|\"([a-zA-Z0-9_\.]{3,})\"|'([a-zA-Z0-9_\.]{3,})'")
    quoted_terms: set[str] = set()
    for m in code_span_regex.finditer(issue_text):
        for g in m.groups():
            if g and len(g.strip()) > 2:
                quoted_terms.add(g.strip())

    # 4. General code identifiers (CamelCase or snake_case)
    id_regex = re.compile(r"\b([A-Za-z][A-Za-z0-9_]{2,})\b")
    all_identifiers = set(id_regex.findall(issue_text))

    # Stopwords filter
    stopwords = {
        "the", "and", "for", "with", "this", "that", "from", "when", "what", "where",
        "which", "error", "failed", "failure", "issue", "bug", "fix", "test", "tests",
        "expected", "received", "actual", "return", "should", "could", "would", "cannot",
        "true", "false", "none", "null", "undefined", "import", "class", "function",
        "line", "file", "traceback", "recent", "call", "last", "most",
    }
    distinctive_ids = {i for i in all_identifiers if i.lower() not in stopwords}

    return {
        "explicit_paths": explicit_paths,
        "error_names": error_names,
        "quoted_terms": quoted_terms,
        "distinctive_ids": distinctive_ids,
        "all_words": {w.lower() for w in distinctive_ids},
    }


def rank_relevant_files(
    arg1: Any,
    arg2: Any,
    arg3: Any = None,
    *,
    issue_text: str | None = None,
    repo_map: RepoMap | None = None,
    symbol_index: SymbolIndex | None = None,
    top_n: int = 15,
) -> list[RankedFile]:
    """Score and rank repository files by relevance to an issue or task description.

    Combines:
      - Stack trace paths and direct file path mentions
      - Symbol index exact and partial matches (functions, classes, methods)
      - Error string / exception matches
      - Code search matches via ripgrep
      - Keyword matching against file paths

    Supports both signatures:
      - ``rank_relevant_files(repo_map, symbol_index, issue_text, top_n=15)``
      - ``rank_relevant_files(issue_text, repo_map, symbol_index, top_n=15)``

    Args:
        arg1: Either RepoMap or issue_text string.
        arg2: Either SymbolIndex or RepoMap.
        arg3: Either issue_text string or SymbolIndex.
        issue_text: Explicit issue_text if using kwargs.
        repo_map: Explicit repo_map if using kwargs.
        symbol_index: Explicit symbol_index if using kwargs.
        top_n: Maximum number of ranked files to return.

    Returns:
        Files sorted by descending relevance score, capped at top_n.
    """
    # Normalize polymorphic arguments
    rm: RepoMap
    si: SymbolIndex
    text: str

    if isinstance(arg1, RepoMap):
        rm = arg1
        si = arg2 if isinstance(arg2, SymbolIndex) else SymbolIndex()
        text = str(arg3 or issue_text or "")
    elif isinstance(arg1, str):
        text = arg1
        rm = arg2 if isinstance(arg2, RepoMap) else (repo_map or RepoMap(root=Path.cwd()))
        si = arg3 if isinstance(arg3, SymbolIndex) else (symbol_index or SymbolIndex())
    else:
        rm = repo_map or RepoMap(root=Path.cwd())
        si = symbol_index or SymbolIndex()
        text = issue_text or ""

    clues = _extract_issue_clues(text)
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    file_paths = [fi.path for fi in rm.files]

    # Pre-run ripgrep code search on top distinctive terms for high-value signal
    search_hits_per_file: dict[str, int] = {}
    top_search_terms = list(clues["quoted_terms"] | clues["error_names"])
    if not top_search_terms and clues["distinctive_ids"]:
        top_search_terms = list(clues["distinctive_ids"])[:4]

    for term in top_search_terms[:4]:
        try:
            hits = search_code(term, path=rm.root, max_results=20)
            for hit in hits:
                # hit.file is relative
                search_hits_per_file[hit.file] = search_hits_per_file.get(hit.file, 0) + 1
        except Exception:
            pass

    for file_path in file_paths:
        file_score = 0.0
        file_reasons: list[str] = []
        p = Path(file_path)
        filename = p.name
        stem = p.stem.lower()

        # 1. Stack trace / explicit path match (+60 to +100)
        for ep in clues["explicit_paths"]:
            if file_path == ep or file_path.endswith(f"/{ep}") or ep.endswith(file_path):
                file_score += 100.0
                file_reasons.append(f"Direct match in stack trace / path '{ep}'")
                break
            elif filename == Path(ep).name:
                file_score += 60.0
                file_reasons.append(f"Filename match '{filename}' in issue path")
                break

        # 2. File basename mentioned directly in text (+40)
        if filename in text:
            file_score += 40.0
            file_reasons.append(f"File '{filename}' explicitly mentioned")

        # 3. Path keywords match (+10 per word in stem/dir)
        path_segments = {seg.lower() for seg in p.parts}
        matched_words = path_segments.intersection(clues["all_words"])
        if matched_words:
            pts = min(len(matched_words) * 12.0, 36.0)
            file_score += pts
            file_reasons.append(f"Path keyword match: {', '.join(sorted(matched_words))}")

        # 4. Symbol matches from SymbolIndex (+25 for exact, +10 for partial)
        symbols = si.index.get(file_path, [])
        matched_symbols = []
        for sym in symbols:
            short_name = sym.name.split(".")[-1]
            if short_name in clues["distinctive_ids"] or short_name in clues["quoted_terms"]:
                matched_symbols.append(sym.name)
                file_score += 30.0
            elif sym.name in clues["distinctive_ids"]:
                matched_symbols.append(sym.name)
                file_score += 35.0
            elif any(err in sym.name for err in clues["error_names"]):
                matched_symbols.append(sym.name)
                file_score += 25.0

        if matched_symbols:
            # Deduplicate symbols while preserving order
            unique_syms = list(dict.fromkeys(matched_symbols))[:3]
            file_reasons.append(f"Symbol match: {', '.join(unique_syms)}")

        # 5. Ripgrep search hits (+15 per hit, max +45)
        if file_path in search_hits_per_file:
            hit_count = search_hits_per_file[file_path]
            file_score += min(hit_count * 15.0, 45.0)
            file_reasons.append(f"Matched {hit_count} code search query hit(s)")

        # 6. Test file adjustment
        is_test_file = "test" in stem or "tests" in path_segments or stem.endswith("_test") or stem.startswith("test_")
        issue_mentions_test = "test" in text.lower()
        if is_test_file and not issue_mentions_test and file_score > 0:
            file_score *= 0.85  # slightly prioritize implementation files unless test is mentioned

        if file_score > 0:
            scores[file_path] = file_score
            reasons[file_path] = file_reasons

    # If nothing scored, fallback to top-level entrypoints and config files
    if not scores:
        for file_path in file_paths:
            p = Path(file_path)
            if p.name in ("main.py", "index.ts", "index.js", "app.py", "main.go", "lib.rs", "package.json", "pyproject.toml"):
                scores[file_path] = 10.0
                reasons[file_path] = ["Default primary entrypoint / config file"]

    # Sort files by descending score
    ranked_tuples = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    results: list[RankedFile] = []

    for path_str, score in ranked_tuples[:top_n]:
        reason_str = "; ".join(reasons.get(path_str, ["Matched query keywords"]))
        results.append(RankedFile(filepath=path_str, score=round(score, 1), reason=reason_str))

    return results


# ---------------------------------------------------------------------------
# Public API: End-to-end explore wrapper
# ---------------------------------------------------------------------------

def explore(
    repo_root: Path | str,
    task: str,
    top_n: int = 15,
) -> ExploreResult:
    """Run the full Explore phase end-to-end.

    Convenience wrapper that calls :func:`build_repo_map`,
    :func:`build_symbol_index`, and :func:`rank_relevant_files` in sequence.

    Args:
        repo_root: Absolute or relative path to the repository root.
        task:      Natural-language task or issue description used for ranking.
        top_n:     Maximum number of ranked files to return.

    Returns:
        An :class:`ExploreResult` aggregating repo_map, symbol_index, and ranked_files.
    """
    root = Path(repo_root).resolve()
    repo_map = build_repo_map(root)
    symbol_index = build_symbol_index(root, repo_map=repo_map)
    ranked_files = rank_relevant_files(repo_map, symbol_index, task, top_n=top_n)

    return ExploreResult(
        repo_map=repo_map,
        symbol_index=symbol_index,
        ranked_files=ranked_files,
    )
