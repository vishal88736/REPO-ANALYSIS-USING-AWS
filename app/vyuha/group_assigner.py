"""
Stage 2: Group Assignment — pure Python.
Every node gets assigned to exactly one group.
"""

import logging
from collections import defaultdict

from app.vyuha.models import DiagramNode, Group
from app.vyuha.node_classifier import GROUP_COLORS

logger = logging.getLogger(__name__)


# Domain → Group mapping
DOMAIN_TO_GROUP = {
    "api":      ("grp_api",       "API Layer",        1),
    "auth":     ("grp_auth",      "Auth / Security",  1),
    "core":     ("grp_core",      "Core Services",    2),
    "billing":  ("grp_domain",    "Domain Logic",     2),
    "worker":   ("grp_workers",   "Workers",          2),
    "data":     ("grp_data",      "Data Layer",       3),
    "infra":    ("grp_infra",     "Infrastructure",   3),
    "external": ("grp_external",  "External",         3),
}


def assign_groups(
    included_nodes: list[DiagramNode],
    external_nodes: list[DiagramNode],
) -> list[Group]:
    """
    Assign every node to a group. Returns the groups with node_ids populated.
    Rules:
      - Every node in exactly one group
      - No group with 1 node (merge into nearest)
      - No group with >8 nodes (split by sublabel/domain)
    """
    all_nodes = included_nodes + external_nodes

    # Step 1: Initial assignment by domain
    group_members: dict[str, list[str]] = defaultdict(list)

    for node in all_nodes:
        domain = node.domain or "core"
        group_key = DOMAIN_TO_GROUP.get(domain, ("grp_core", "Core Services", 2))[0]

        # Override: entry points always in API Layer
        if node.is_entry_point:
            group_key = "grp_api"
        # Override: externals always in External
        elif node.node_type in ("external", "cloud"):
            group_key = "grp_external"
        elif node.node_type == "database":
            group_key = "grp_data"
        elif node.node_type == "queue":
            group_key = "grp_infra"

        group_members[group_key].append(node.id)

    # Step 2: Merge groups with 1 node
    to_merge = [gk for gk, members in group_members.items() if len(members) == 1]
    for gk in to_merge:
        orphan_id = group_members[gk][0]
        # Find nearest group (same layer_order or +1)
        current_layer = DOMAIN_TO_GROUP.get(
            next((d for d, (g, _, _) in DOMAIN_TO_GROUP.items() if g == gk), "core"),
            ("grp_core", "Core Services", 2)
        )[2]

        best_target = None
        best_size = 999
        for other_gk, other_members in group_members.items():
            if other_gk == gk:
                continue
            other_layer = DOMAIN_TO_GROUP.get(
                next((d for d, (g, _, _) in DOMAIN_TO_GROUP.items() if g == other_gk), "core"),
                ("grp_core", "Core Services", 2)
            )[2]
            if abs(other_layer - current_layer) <= 1 and len(other_members) < best_size:
                best_target = other_gk
                best_size = len(other_members)

        if best_target:
            group_members[best_target].append(orphan_id)
        else:
            # No good target — merge into Core Services
            group_members.setdefault("grp_core", []).append(orphan_id)

        del group_members[gk]

    # Step 3: Split groups with >8 nodes
    to_split = [(gk, members) for gk, members in group_members.items() if len(members) > 8]
    for gk, members in to_split:
        # Split in half
        mid = len(members) // 2
        group_members[gk] = members[:mid]
        new_gk = f"{gk}_2"
        group_members[new_gk] = members[mid:]

    # Step 4: Build Group objects
    groups = []
    for gk, member_ids in group_members.items():
        if not member_ids:
            continue

        # Find group metadata
        base_gk = gk.rstrip("_2")
        group_info = None
        for domain, (g, label, layer) in DOMAIN_TO_GROUP.items():
            if g == base_gk:
                group_info = (label, layer)
                break

        if not group_info:
            group_info = ("Core Services", 2)

        label, layer = group_info
        if gk.endswith("_2"):
            label = f"{label} (cont.)"

        color = GROUP_COLORS.get(label.rstrip(" (cont.)"), "#94A3B8")

        groups.append(Group(
            id=gk,
            label=label,
            description=f"Contains {len(member_ids)} components",
            color=color,
            node_ids=member_ids,
            layer_order=layer,
        ))

    groups.sort(key=lambda g: g.layer_order)

    logger.info("Group assignment: %d groups from %d nodes", len(groups), len(all_nodes))
    return groups