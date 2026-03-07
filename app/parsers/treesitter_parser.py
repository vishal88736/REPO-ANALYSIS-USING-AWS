"""
Tree-sitter based static code parser.

Supports: Python, JavaScript, TypeScript, Go, Java, Rust, C, C++.
Extracts: functions, classes, methods, imports.
"""

import logging
import importlib
from typing import Any

from tree_sitter import Language, Parser, Node

from app.schemas.analysis import ParsedStructure, ParsedFunction, ParsedClass, ParsedImport

logger = logging.getLogger(__name__)

LANGUAGE_MAP: dict[str, str] = {
    ".py": "tree_sitter_python",
    ".js": "tree_sitter_javascript",
    ".jsx": "tree_sitter_javascript",
    ".ts": "tree_sitter_typescript",
    ".tsx": "tree_sitter_typescript",
    ".go": "tree_sitter_go",
    ".java": "tree_sitter_java",
    ".rs": "tree_sitter_rust",
    ".c": "tree_sitter_c",
    ".cpp": "tree_sitter_cpp",
    ".cc": "tree_sitter_cpp",
    ".h": "tree_sitter_c",
    ".hpp": "tree_sitter_cpp",
}

_parser_cache: dict[str, Parser] = {}


def _get_parser(extension: str) -> Parser | None:
    if extension in _parser_cache:
        return _parser_cache[extension]

    module_name = LANGUAGE_MAP.get(extension)
    if not module_name:
        return None

    try:
        mod = importlib.import_module(module_name)
        lang = Language(mod.language())
        parser = Parser(lang)
        _parser_cache[extension] = parser
        return parser
    except ImportError:
        logger.warning("Tree-sitter grammar not installed: %s", module_name)
        return None
    except Exception as e:
        logger.warning("Failed to init tree-sitter for %s: %s", extension, e)
        return None


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_python(root: Node, source: bytes) -> ParsedStructure:
    functions, classes, imports = [], [], []

    for child in root.children:
        if child.type == "function_definition":
            name_node = child.child_by_field_name("name")
            params_node = child.child_by_field_name("parameters")
            params = []
            if params_node:
                for p in params_node.children:
                    if p.type == "identifier":
                        params.append(_text(p, source))
            functions.append(ParsedFunction(
                name=_text(name_node, source) if name_node else "unknown",
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                parameters=params,
            ))

        elif child.type == "class_definition":
            name_node = child.child_by_field_name("name")
            class_name = _text(name_node, source) if name_node else "unknown"
            methods = []
            body = child.child_by_field_name("body")
            if body:
                for stmt in body.children:
                    if stmt.type == "function_definition":
                        m = stmt.child_by_field_name("name")
                        if m:
                            methods.append(_text(m, source))
            classes.append(ParsedClass(
                name=class_name,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                methods=methods,
            ))

        elif child.type == "import_statement":
            txt = _text(child, source)
            parts = txt.replace("import ", "").strip().split(",")
            imports.append(ParsedImport(module=parts[0].strip(), names=[p.strip() for p in parts]))

        elif child.type == "import_from_statement":
            txt = _text(child, source)
            if " import " in txt:
                mod_part = txt.split(" import ")[0].replace("from ", "").strip()
                names_part = txt.split(" import ")[1].strip()
                names = [n.strip() for n in names_part.split(",")]
                imports.append(ParsedImport(module=mod_part, names=names))

    return ParsedStructure(functions=functions, classes=classes, imports=imports)


def _extract_js_ts(root: Node, source: bytes) -> ParsedStructure:
    functions, classes, imports = [], [], []

    def walk(node: Node):
        if node.type in ("function_declaration", "method_definition"):
            name_node = node.child_by_field_name("name")
            params_node = node.child_by_field_name("parameters")
            params = []
            if params_node:
                for p in params_node.children:
                    if p.type in ("identifier", "required_parameter", "optional_parameter"):
                        params.append(_text(p, source))
            functions.append(ParsedFunction(
                name=_text(name_node, source) if name_node else "anonymous",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parameters=params,
            ))

        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            class_name = _text(name_node, source) if name_node else "unknown"
            methods = []
            body = node.child_by_field_name("body")
            if body:
                for stmt in body.children:
                    if stmt.type == "method_definition":
                        m = stmt.child_by_field_name("name")
                        if m:
                            methods.append(_text(m, source))
            classes.append(ParsedClass(
                name=class_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                methods=methods,
            ))

        elif node.type == "import_statement":
            source_node = node.child_by_field_name("source")
            module = _text(source_node, source).strip("'\"") if source_node else _text(node, source)
            imports.append(ParsedImport(module=module, names=[]))

        for c in node.children:
            walk(c)

    walk(root)
    return ParsedStructure(functions=functions, classes=classes, imports=imports)


def _extract_generic(root: Node, source: bytes) -> ParsedStructure:
    functions, classes = [], []

    def walk(node: Node):
        if node.type in ("function_declaration", "function_definition",
                         "method_declaration", "function_item"):
            name_node = node.child_by_field_name("name")
            functions.append(ParsedFunction(
                name=_text(name_node, source) if name_node else "unknown",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))

        elif node.type in ("class_declaration", "struct_item", "struct_specifier", "impl_item"):
            name_node = node.child_by_field_name("name")
            classes.append(ParsedClass(
                name=_text(name_node, source) if name_node else "unknown",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))

        for c in node.children:
            walk(c)

    walk(root)
    return ParsedStructure(functions=functions, classes=classes)


_EXTRACTORS: dict[str, Any] = {
    ".py": _extract_python,
    ".js": _extract_js_ts, ".jsx": _extract_js_ts,
    ".ts": _extract_js_ts, ".tsx": _extract_js_ts,
}


def parse_file(file_path: str, content: bytes, extension: str) -> ParsedStructure:
    """Parse a source file using tree-sitter and extract structure."""
    parser = _get_parser(extension)
    if parser is None:
        return ParsedStructure()

    try:
        tree = parser.parse(content)
        extractor = _EXTRACTORS.get(extension, _extract_generic)
        result = extractor(tree.root_node, content)
        logger.debug("Parsed %s: %d funcs, %d classes", file_path, len(result.functions), len(result.classes))
        return result
    except Exception as e:
        logger.warning("Tree-sitter parsing failed for %s: %s", file_path, e)
        return ParsedStructure()