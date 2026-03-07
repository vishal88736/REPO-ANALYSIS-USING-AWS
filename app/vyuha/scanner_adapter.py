"""
Adapter: converts existing FileAnalysisResult data into ParsedRepo format.
Ensures consistent node IDs between nodes and edges.
"""

import logging
from datetime import datetime, timezone

from app.schemas.analysis import FileAnalysisResult, ArchitectureSummary
from app.vyuha.scanner_models import (
    ParsedRepo, RepoMeta, ParsedNode, ParsedEdge,
    Package, EntryPoint, ExternalDep,
)

logger = logging.getLogger(__name__)


def file_analyses_to_parsed_repo(
    repo_name: str,
    repo_url: str,
    file_analyses: list[FileAnalysisResult],
    architecture: ArchitectureSummary | None = None,
) -> ParsedRepo:
    """Convert FileAnalysisResult list to ParsedRepo with consistent IDs."""
    nodes = []
    edges = []
    packages = []
    entry_points = []
    external_deps_map: dict[str, ExternalDep] = {}

    languages = set()
    lang_map = {
        "js": "JavaScript", "ts": "TypeScript", "py": "Python",
        "go": "Go", "java": "Java", "rs": "Rust",
        "json": "JSON", "md": "Markdown", "css": "CSS", "html": "HTML",
    }

    # Track ALL node IDs for edge resolution
    all_node_ids: dict[str, str] = {}  # various keys → canonical node ID
    edge_counter = 0

    for fa in file_analyses:
        ext = fa.file_path.split(".")[-1] if "." in fa.file_path else ""
        lang = lang_map.get(ext, ext.upper() if ext else "Unknown")
        languages.add(lang)

        # === File node (always created) ===
        file_node_id = f"file:{fa.file_path}"
        nodes.append(ParsedNode(
            id=file_node_id,
            kind="module",  # Use "module" not "file" so it passes classification
            name=fa.file_path.split("/")[-1],
            qualified_name=fa.file_path,
            file=fa.file_path,
            language=lang,
            is_exported=True,
            description=fa.summary[:200] if fa.summary else "",
        ))

        # Register in lookup (multiple keys for the same node)
        all_node_ids[file_node_id] = file_node_id
        all_node_ids[fa.file_path] = file_node_id
        all_node_ids[fa.file_path.split("/")[-1]] = file_node_id

        func_node_ids = [file_node_id]

        # === Function nodes ===
        for func in fa.functions:
            func_node_id = f"func:{fa.file_path}:{func.name}"
            nodes.append(ParsedNode(
                id=func_node_id,
                kind="function",
                name=func.name,
                qualified_name=f"{fa.file_path}::{func.name}",
                file=fa.file_path,
                language=lang,
                is_exported=True,
                description=func.description,
            ))
            func_node_ids.append(func_node_id)

            all_node_ids[func_node_id] = func_node_id
            all_node_ids[func.name] = func_node_id

            # Detect entry points
            if func.name in ("main", "__main__", "app", "run", "start", "handler", "init"):
                entry_points.append(EntryPoint(
                    node_id=func_node_id,
                    kind="main" if func.name in ("main", "__main__") else "http_handler",
                    description=func.description,
                ))

            # Function call edges
            for called in func.calls:
                edge_counter += 1
                edges.append(ParsedEdge(
                    id=f"edge_{edge_counter}",
                    kind="calls",
                    source_id=func_node_id,
                    target_id=f"unresolved:{called}",
                    is_resolved=False,
                ))

        # === Class nodes ===
        for cls in fa.classes:
            cls_node_id = f"class:{fa.file_path}:{cls.name}"
            nodes.append(ParsedNode(
                id=cls_node_id,
                kind="class",
                name=cls.name,
                qualified_name=f"{fa.file_path}::{cls.name}",
                file=fa.file_path,
                language=lang,
                is_exported=True,
            ))
            func_node_ids.append(cls_node_id)
            all_node_ids[cls_node_id] = cls_node_id
            all_node_ids[cls.name] = cls_node_id

        # === Package ===
        packages.append(Package(
            id=f"pkg:{fa.file_path}",
            name=fa.file_path.split("/")[-1],
            import_path=fa.file_path,
            files=[fa.file_path],
            node_ids=func_node_ids,
            language=lang,
            is_internal=True,
        ))

        # === External deps ===
        for dep in fa.external_dependencies:
            if dep not in external_deps_map:
                external_deps_map[dep] = ExternalDep(
                    name=dep,
                    import_path=dep,
                    category=_guess_dep_category(dep),
                    used_by=[],
                )
            external_deps_map[dep].used_by.append(file_node_id)

        # === File interaction edges (MOST IMPORTANT for architecture) ===
        for inter in fa.file_interactions:
            edge_counter += 1
            src_id = f"file:{inter.source_file}"
            tgt_id = f"file:{inter.target_file}"
            edges.append(ParsedEdge(
                id=f"edge_{edge_counter}",
                kind="depends_on" if inter.interaction_type == "imports" else "calls",
                source_id=src_id,
                target_id=tgt_id,
                is_resolved=True,
            ))

        # === Internal file reference edges ===
        for ref in fa.internal_file_references:
            edge_counter += 1
            src_id = f"file:{fa.file_path}"
            tgt_id = f"file:{ref}"
            edges.append(ParsedEdge(
                id=f"edge_{edge_counter}",
                kind="depends_on",
                source_id=src_id,
                target_id=tgt_id,
                is_resolved=True,
            ))

    # === Resolve unresolved call edges ===
    resolved_edges = []
    for edge in edges:
        if edge.is_resolved:
            resolved_edges.append(edge)
        else:
            called_name = edge.target_id.replace("unresolved:", "")
            # Try to find by function name
            resolved_id = all_node_ids.get(called_name)
            if resolved_id:
                edge.target_id = resolved_id
                edge.is_resolved = True
            resolved_edges.append(edge)

    # === Entry points from architecture summary ===
    if architecture and architecture.entry_points:
        for ep in architecture.entry_points:
            ep_node_id = f"file:{ep.file_path}"
            if ep_node_id in all_node_ids:
                if not any(e.node_id == ep_node_id for e in entry_points):
                    entry_points.append(EntryPoint(
                        node_id=ep_node_id,
                        kind="main",
                        description=ep.reason,
                    ))

    # === Fallback entry points ===
    if not entry_points:
        priority_files = ["manifest.json", "main.py", "main.go", "index.js", "index.ts", "app.py", "app.js"]
        for pf in priority_files:
            for fa in file_analyses:
                if fa.file_path == pf or fa.file_path.endswith(f"/{pf}"):
                    entry_points.append(EntryPoint(
                        node_id=f"file:{fa.file_path}",
                        kind="main",
                        description=f"Primary entry: {fa.file_path}",
                    ))
                    break

    # === Deduplicate edges ===
    seen_edges = set()
    deduped = []
    for edge in resolved_edges:
        key = (edge.source_id, edge.target_id, edge.kind)
        if key not in seen_edges:
            seen_edges.add(key)
            deduped.append(edge)

    clean_languages = sorted(languages - {"JSON", "Markdown", "CSS", "HTML", "Unknown"})

    logger.info(
        "Adapter: %d files → %d nodes, %d edges, %d entry_points, %d ext_deps",
        len(file_analyses), len(nodes), len(deduped), len(entry_points), len(external_deps_map),
    )

    return ParsedRepo(
        meta=RepoMeta(
            repo_name=repo_name,
            repo_url=repo_url,
            languages=clean_languages,
            total_files=len(file_analyses),
            total_nodes=len(nodes),
            scanned_at=datetime.now(timezone.utc).isoformat(),
        ),
        nodes=nodes,
        edges=deduped,
        packages=packages,
        entry_points=entry_points,
        external_deps=list(external_deps_map.values()),
    )


def _guess_dep_category(dep_name: str) -> str:
    dep_lower = dep_name.lower()
    categories = {
        "db": ["postgres", "mysql", "sqlite", "mongo", "sqlalchemy", "prisma", "sequelize", "knex"],
        "cache": ["redis", "memcache"],
        "queue": ["kafka", "rabbitmq", "celery", "bull", "sqs"],
        "cloud": ["aws", "boto", "gcp", "azure", "s3"],
        "auth": ["jwt", "oauth", "passport", "bcrypt"],
        "payment": ["stripe", "razorpay", "braintree", "paypal"],
        "http": ["axios", "requests", "fetch", "httpx", "got"],
        "pdf": ["jspdf", "pdfkit", "reportlab", "puppeteer"],
    }
    for cat, keywords in categories.items():
        if any(kw in dep_lower for kw in keywords):
            return cat
    return "library"