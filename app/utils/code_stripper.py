"""
Code Stripper — removes comments, blank lines, docstrings for 40+ languages.
Each language has its own comment syntax handled correctly.
Preserves string literals (won't strip // inside "https://url").
"""

import re
from typing import Callable


# ═══════════════════════════════════════════════════════════════════
# EXTENSION → LANGUAGE MAPPING
# ═══════════════════════════════════════════════════════════════════

EXTENSION_MAP: dict[str, str] = {
    # Python
    "py": "python", "pyw": "python", "pyi": "python",
    "pyx": "python", "pxd": "python",

    # JavaScript / TypeScript
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "jsx": "javascript",
    "ts": "typescript", "tsx": "typescript", "mts": "typescript",

    # Go
    "go": "go",

    # Java / Kotlin / Scala / Groovy
    "java": "java",
    "kt": "kotlin", "kts": "kotlin",
    "scala": "scala", "sc": "scala",
    "groovy": "groovy", "gradle": "groovy",

    # C / C++ / Objective-C
    "c": "c", "h": "c",
    "cpp": "cpp", "cc": "cpp", "cxx": "cpp",
    "hpp": "cpp", "hh": "cpp", "hxx": "cpp",
    "m": "objc", "mm": "objc",

    # C# / F#
    "cs": "csharp",
    "fs": "fsharp", "fsx": "fsharp",

    # Rust
    "rs": "rust",

    # Swift
    "swift": "swift",

    # Dart / Flutter
    "dart": "dart",

    # Ruby
    "rb": "ruby", "rake": "ruby", "gemspec": "ruby",
    "ru": "ruby",

    # PHP
    "php": "php", "phtml": "php",

    # Perl
    "pl": "perl", "pm": "perl", "t": "perl",

    # Lua
    "lua": "lua",

    # R
    "r": "r", "R": "r", "rmd": "r",

    # Julia
    "jl": "julia",

    # Elixir / Erlang
    "ex": "elixir", "exs": "elixir",
    "erl": "erlang", "hrl": "erlang",

    # Haskell
    "hs": "haskell", "lhs": "haskell",

    # Clojure
    "clj": "clojure", "cljs": "clojure", "cljc": "clojure",
    "edn": "clojure",

    # OCaml
    "ml": "ocaml", "mli": "ocaml",

    # Zig
    "zig": "zig",

    # Nim
    "nim": "nim", "nims": "nim",

    # V
    "v": "vlang",

    # Shell / Bash
    "sh": "shell", "bash": "shell", "zsh": "shell",
    "fish": "shell", "ksh": "shell",

    # PowerShell
    "ps1": "powershell", "psm1": "powershell", "psd1": "powershell",

    # SQL
    "sql": "sql",

    # HTML / XML / SVG
    "html": "html", "htm": "html",
    "xml": "xml", "xsl": "xml", "xslt": "xml",
    "svg": "xml",
    "vue": "vue",
    "svelte": "svelte",

    # CSS / SCSS / LESS / SASS
    "css": "css",
    "scss": "scss", "sass": "sass",
    "less": "less",
    "styl": "stylus",

    # Config / Data
    "yaml": "yaml", "yml": "yaml",
    "toml": "toml",
    "ini": "ini", "cfg": "ini",
    "conf": "ini",
    "env": "dotenv",
    "properties": "properties",

    # JSON (no comments by default, but JSONC has //)
    "json": "json", "jsonc": "jsonc",
    "json5": "jsonc",

    # Markdown / Text
    "md": "markdown", "mdx": "markdown",
    "rst": "rst",
    "txt": "text",

    # Protobuf / Thrift / GraphQL
    "proto": "protobuf",
    "thrift": "thrift",
    "graphql": "graphql", "gql": "graphql",

    # Terraform / HCL
    "tf": "hcl", "hcl": "hcl",
    "tfvars": "hcl",

    # Dockerfile
    "dockerfile": "dockerfile",

    # Makefile
    "mk": "makefile",

    # WASM / Assembly
    "asm": "assembly", "s": "assembly",
    "wat": "wasm",

    # Solidity
    "sol": "solidity",

    # Nix
    "nix": "nix",
}

# Special filenames without extensions
FILENAME_MAP: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
    "GNUmakefile": "makefile",
    "Rakefile": "ruby",
    "Gemfile": "ruby",
    "Vagrantfile": "ruby",
    "Jenkinsfile": "groovy",
    "BUILD": "starlark",
    "WORKSPACE": "starlark",
    "BUILD.bazel": "starlark",
    "WORKSPACE.bazel": "starlark",
    "CMakeLists.txt": "cmake",
    ".gitignore": "gitignore",
    ".dockerignore": "gitignore",
    ".editorconfig": "ini",
    ".eslintrc": "json",
    ".prettierrc": "json",
    "nginx.conf": "nginx",
    "httpd.conf": "apache",
}


# ═══════════════════════════════════════════════════════════════════
# STRING-AWARE HELPERS (won't break strings containing comment chars)
# ═══════════════════════════════════════════════════════════════════

def _find_outside_strings(line: str, target: str, quotes: str = "\"'`") -> int:
    """Find target substring outside of string literals. Returns -1 if not found."""
    in_char = None
    escaped = False
    i = 0
    while i < len(line):
        ch = line[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if in_char is None:
            if ch in quotes:
                in_char = ch
            elif line[i:i + len(target)] == target:
                return i
        else:
            if ch == in_char:
                in_char = None
        i += 1
    return -1


def _remove_inline(line: str, marker: str, quotes: str = "\"'`") -> str:
    """Remove inline comment starting with marker, respecting strings."""
    pos = _find_outside_strings(line, marker, quotes)
    if pos >= 0:
        return line[:pos].rstrip()
    return line


# ═══════════════════════════════════════════════════════════════════
# LANGUAGE-SPECIFIC STRIPPERS
# ═══════════════════════════════════════════════════════════════════

def _strip_blank_lines(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if line.strip())


# ── Python ────────────────────────────────────────────────────────
def _strip_python(content: str) -> str:
    """Strip # comments, docstrings (''' and \"\"\"), blank lines."""
    result = []
    lines = content.split("\n")
    in_docstring = False
    doc_quote = None
    for line in lines:
        stripped = line.strip()

        # Inside docstring — skip until closing
        if in_docstring:
            if doc_quote in stripped:
                in_docstring = False
            continue

        # Docstring start (triple quote on its own or at line start)
        if not in_docstring:
            for q in ('"""', "'''"):
                if stripped.startswith(q):
                    # Single-line docstring: """text"""
                    if stripped.count(q) >= 2 and stripped.endswith(q) and len(stripped) > 3:
                        break  # skip this line
                    # Multi-line start
                    if stripped.count(q) == 1:
                        in_docstring = True
                        doc_quote = q
                        break
                    # Single-line: """text"""
                    break
            else:
                # No docstring — process normally
                if not stripped or stripped.startswith("#"):
                    continue
                line = _remove_inline(line, "#", "\"'")
                if line.strip():
                    result.append(line)
                continue
            continue  # was a docstring line

    return "\n".join(result)


# ── C-style (JS, TS, Go, Java, C, C++, Rust, Kotlin, Swift, etc.) ─
def _strip_c_style(content: str) -> str:
    """Strip // line comments, /* block comments */, blank lines."""
    # Remove block comments first
    content = re.sub(r'/\*[\s\S]*?\*/', '', content)
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            continue
        line = _remove_inline(line, "//", "\"'`")
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ── Ruby ──────────────────────────────────────────────────────────
def _strip_ruby(content: str) -> str:
    """Strip # comments, =begin/=end blocks, blank lines."""
    content = re.sub(r'^=begin\s*\n[\s\S]*?^=end\s*\n?', '', content, flags=re.MULTILINE)
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        line = _remove_inline(line, "#", "\"'")
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ── Perl ──────────────────────────────────────────────────────────
def _strip_perl(content: str) -> str:
    """Strip # comments, __END__/=pod blocks, blank lines."""
    # Remove __END__ section
    end_idx = content.find("\n__END__\n")
    if end_idx >= 0:
        content = content[:end_idx]
    # Remove =pod ... =cut blocks
    content = re.sub(r'^=pod\s*\n[\s\S]*?^=cut\s*\n?', '', content, flags=re.MULTILINE)
    return _strip_hash(content)


# ── Lua ───────────────────────────────────────────────────────────
def _strip_lua(content: str) -> str:
    """Strip -- line comments, --[[ block comments ]], blank lines."""
    content = re.sub(r'--\[\[[\s\S]*?\]\]', '', content)
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("--"):
            continue
        line = _remove_inline(line, "--", "\"'")
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ── R ─────────────────────────────────────────────────────────────
def _strip_r(content: str) -> str:
    """Strip # comments, blank lines."""
    return _strip_hash(content)


# ── Julia ─────────────────────────────────────────────────────────
def _strip_julia(content: str) -> str:
    """Strip # line comments, #= block comments =#, blank lines."""
    content = re.sub(r'#=[\s\S]*?=#', '', content)
    return _strip_hash(content)


# ── Elixir ────────────────────────────────────────────────────────
def _strip_elixir(content: str) -> str:
    """Strip # comments, @moduledoc/@doc heredocs, blank lines."""
    # Remove @moduledoc and @doc heredocs
    content = re.sub(r'@(?:moduledoc|doc)\s+"""[\s\S]*?"""', '', content)
    content = re.sub(r"@(?:moduledoc|doc)\s+'''[\s\S]*?'''", '', content)
    return _strip_hash(content)


# ── Erlang ────────────────────────────────────────────────────────
def _strip_erlang(content: str) -> str:
    """Strip % comments, blank lines."""
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        line = _remove_inline(line, "%", "\"'")
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ── Haskell ───────────────────────────────────────────────────────
def _strip_haskell(content: str) -> str:
    """Strip -- line comments, {- block comments -}, blank lines."""
    content = re.sub(r'\{-[\s\S]*?-\}', '', content)
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("--"):
            continue
        line = _remove_inline(line, "--", "\"'")
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ── Clojure ───────────────────────────────────────────────────────
def _strip_clojure(content: str) -> str:
    """Strip ; comments, blank lines."""
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        pos = _find_outside_strings(line, ";", "\"")
        if pos >= 0:
            line = line[:pos].rstrip()
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ── OCaml ───────────────────────────────────────────────��─────────
def _strip_ocaml(content: str) -> str:
    """Strip (* block comments *), blank lines."""
    content = re.sub(r'\(\*[\s\S]*?\*\)', '', content)
    return _strip_blank_lines(content)


# ── Zig ───────────────────────────────────────────────────────────
def _strip_zig(content: str) -> str:
    """Strip // comments, blank lines. (Zig has no block comments.)"""
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        line = _remove_inline(line, "//", "\"'")
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ── Nim ───────────────────────────────────────────────────────────
def _strip_nim(content: str) -> str:
    """Strip # line comments, #[ block comments ]#, blank lines."""
    content = re.sub(r'#\[[\s\S]*?\]#', '', content)
    return _strip_hash(content)


# ── V lang ────────────────────────────────────────────────────────
def _strip_vlang(content: str) -> str:
    """Strip // comments, /* block */, blank lines."""
    return _strip_c_style(content)


# ── PowerShell ────────────────────────────────────────────────��───
def _strip_powershell(content: str) -> str:
    """Strip # line comments, <# block comments #>, blank lines."""
    content = re.sub(r'<#[\s\S]*?#>', '', content)
    return _strip_hash(content)


# ── SQL ───────────────────────────────────────────────────────────
def _strip_sql(content: str) -> str:
    """Strip -- line comments, /* block comments */, blank lines."""
    content = re.sub(r'/\*[\s\S]*?\*/', '', content)
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        line = _remove_inline(line, "--", "\"'")
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ── HTML / XML / Vue / Svelte ─────────────────────────────────────
def _strip_html(content: str) -> str:
    """Strip <!-- comments -->, blank lines."""
    content = re.sub(r'<!--[\s\S]*?-->', '', content)
    return _strip_blank_lines(content)


# ── CSS / SCSS / LESS ─────────────────────────────────────────────
def _strip_css(content: str) -> str:
    """Strip /* block comments */, // line comments (SCSS/LESS), blank lines."""
    content = re.sub(r'/\*[\s\S]*?\*/', '', content)
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            continue
        result.append(line)
    return "\n".join(result)


# ── SASS (indented) ──────────────────────────────────────────────
def _strip_sass(content: str) -> str:
    """Strip // line comments, /* block comments */, blank lines."""
    return _strip_css(content)


# ── HCL / Terraform ──────────────────────────────────────────────
def _strip_hcl(content: str) -> str:
    """Strip # and // line comments, /* block */, blank lines."""
    content = re.sub(r'/\*[\s\S]*?\*/', '', content)
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        line = _remove_inline(line, "#", "\"'")
        line = _remove_inline(line, "//", "\"'")
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ── Protobuf / Thrift ────────────────────────────────────────────
def _strip_protobuf(content: str) -> str:
    """Strip // line comments, /* block */, blank lines."""
    return _strip_c_style(content)


# ── GraphQL ───────────────────────────────────────────────────────
def _strip_graphql(content: str) -> str:
    """Strip # comments, blank lines."""
    return _strip_hash(content)


# ── Solidity ──────────────────────────────────────────────────────
def _strip_solidity(content: str) -> str:
    """Strip // and /* */, NatSpec (///), blank lines."""
    content = re.sub(r'/\*[\s\S]*?\*/', '', content)
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        line = _remove_inline(line, "//", "\"'")
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ── Nix ───────────────────────────────────────────────────────────
def _strip_nix(content: str) -> str:
    """Strip # line comments, /* block */, blank lines."""
    content = re.sub(r'/\*[\s\S]*?\*/', '', content)
    return _strip_hash(content)


# ── CMake ─────────────────────────────────────────────────────────
def _strip_cmake(content: str) -> str:
    """Strip # comments, #[[ block comments ]], blank lines."""
    content = re.sub(r'#\[\[[\s\S]*?\]\]', '', content)
    return _strip_hash(content)


# ── Starlark (Bazel BUILD files) ─────────────────────────────────
def _strip_starlark(content: str) -> str:
    """Strip # comments, blank lines. Same as Python-style."""
    return _strip_hash(content)


# ── Nginx / Apache config ────────────────────────────────────────
def _strip_nginx(content: str) -> str:
    """Strip # comments, blank lines."""
    return _strip_hash(content)


# ── JSONC (JSON with comments) ────────────────────────────────────
def _strip_jsonc(content: str) -> str:
    """Strip // and /* */ comments from JSONC/JSON5."""
    return _strip_c_style(content)


# ── Markdown / RST ────────────────────────────────────────────────
def _strip_markdown(content: str) -> str:
    """Strip blank lines. (Markdown has no real comments to strip.)"""
    return _strip_blank_lines(content)


# ── Dockerfile ────────────────────────────────────────────────────
def _strip_dockerfile(content: str) -> str:
    """Strip # comments, blank lines."""
    return _strip_hash(content)


# ── Makefile ──────────────────────────────────────────────────────
def _strip_makefile(content: str) -> str:
    """Strip # comments, blank lines. (Careful: \t matters in Makefiles.)"""
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Don't strip inline # in Makefiles — could be part of shell commands
        result.append(line)
    return "\n".join(result)


# ── .gitignore / .dockerignore ────────────────────────────────────
def _strip_gitignore(content: str) -> str:
    """Strip # comments, blank lines."""
    return _strip_hash(content)


# ── .env / dotenv ─────────────────────────────────────────────────
def _strip_dotenv(content: str) -> str:
    """Strip # comments, blank lines."""
    return _strip_hash(content)


# ── .properties (Java) ───────────────────────────────────────────
def _strip_properties(content: str) -> str:
    """Strip # and ! comments, blank lines."""
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        result.append(line)
    return "\n".join(result)


# ── Assembly ──────────────────────────────────────────────────────
def _strip_assembly(content: str) -> str:
    """Strip ; comments (x86), # comments (ARM/GAS), blank lines."""
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("#"):
            continue
        result.append(line)
    return "\n".join(result)


# ── Generic # comment stripper (reused by many languages) ─────────
def _strip_hash(content: str) -> str:
    """Generic: strip # comments and blank lines."""
    result = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        line = _remove_inline(line, "#", "\"'")
        if line.strip():
            result.append(line)
    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════
# LANGUAGE → STRIPPER DISPATCH TABLE
# ═══════════════════════════════════════════════════════════════════

STRIPPERS: dict[str, Callable[[str], str]] = {
    # Python
    "python": _strip_python,

    # C-style family
    "javascript": _strip_c_style,
    "typescript": _strip_c_style,
    "go": _strip_c_style,
    "java": _strip_c_style,
    "kotlin": _strip_c_style,
    "scala": _strip_c_style,
    "groovy": _strip_c_style,
    "c": _strip_c_style,
    "cpp": _strip_c_style,
    "objc": _strip_c_style,
    "csharp": _strip_c_style,
    "rust": _strip_c_style,
    "swift": _strip_c_style,
    "dart": _strip_c_style,
    "vlang": _strip_vlang,

    # Unique comment syntax
    "ruby": _strip_ruby,
    "php": _strip_c_style,
    "perl": _strip_perl,
    "lua": _strip_lua,
    "r": _strip_r,
    "julia": _strip_julia,
    "elixir": _strip_elixir,
    "erlang": _strip_erlang,
    "haskell": _strip_haskell,
    "clojure": _strip_clojure,
    "ocaml": _strip_ocaml,
    "zig": _strip_zig,
    "nim": _strip_nim,
    "fsharp": _strip_c_style,

    # Shell / Config
    "shell": _strip_hash,
    "powershell": _strip_powershell,
    "dockerfile": _strip_dockerfile,
    "makefile": _strip_makefile,
    "gitignore": _strip_gitignore,
    "dotenv": _strip_dotenv,
    "ini": _strip_hash,
    "properties": _strip_properties,

    # Data / Config
    "yaml": _strip_hash,
    "toml": _strip_hash,
    "json": _strip_blank_lines,  # standard JSON has no comments
    "jsonc": _strip_jsonc,

    # SQL
    "sql": _strip_sql,

    # Web
    "html": _strip_html,
    "xml": _strip_html,
    "vue": _strip_html,
    "svelte": _strip_html,
    "css": _strip_css,
    "scss": _strip_css,
    "sass": _strip_sass,
    "less": _strip_css,
    "stylus": _strip_hash,

    # Infrastructure
    "hcl": _strip_hcl,
    "cmake": _strip_cmake,
    "starlark": _strip_starlark,
    "nginx": _strip_nginx,
    "apache": _strip_hash,
    "nix": _strip_nix,

    # Schema / API
    "protobuf": _strip_protobuf,
    "thrift": _strip_c_style,
    "graphql": _strip_graphql,

    # Blockchain
    "solidity": _strip_solidity,

    # Assembly
    "assembly": _strip_assembly,
    "wasm": _strip_c_style,

    # Docs
    "markdown": _strip_markdown,
    "rst": _strip_blank_lines,
    "text": _strip_blank_lines,
}


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def get_language(file_path: str) -> str:
    """Detect language from file path."""
    filename = file_path.replace("\\", "/").split("/")[-1]

    # Check exact filename first
    if filename in FILENAME_MAP:
        return FILENAME_MAP[filename]

    # Check extension
    ext = filename.split(".")[-1] if "." in filename else ""
    return EXTENSION_MAP.get(ext.lower(), "unknown")


def strip_code(content: str, language: str) -> str:
    """Strip comments, blank lines, docstrings for the given language."""
    stripper = STRIPPERS.get(language.lower(), _strip_blank_lines)
    return stripper(content)


def strip_for_llm(content: str, file_path: str) -> tuple[str, dict]:
    """
    Strip code for LLM consumption.
    Returns (stripped_content, stats_dict).
    """
    lang = get_language(file_path)
    original_lines = content.count("\n") + 1
    original_chars = len(content)

    stripped = strip_code(content, lang)

    stripped_lines = stripped.count("\n") + 1
    stripped_chars = len(stripped)
    saved_pct = round((1 - stripped_chars / max(original_chars, 1)) * 100, 1)

    return stripped, {
        "original_lines": original_lines,
        "stripped_lines": stripped_lines,
        "original_chars": original_chars,
        "stripped_chars": stripped_chars,
        "saved_percent": saved_pct,
        "language": lang,
    }