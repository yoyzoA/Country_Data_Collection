"""
Document Tree Patching

Given:
  - A source tree T1 (as a TreeNode)
  - An edit script ES(T1, T2) produced by ted_diff.compare_trees()

Produces:
  - A patched tree obtained by applying the edit script to a deep copy of T1
  - Ideally this patched tree matches T2, but because the edit script comes
    from a preorder-sequence TED approximation, exact reconstruction is not
    always guaranteed. The result is therefore verified afterwards using
    residual TED against the true target tree.

How it works:
The edit script is a list of node-level operations recovered by backtracking
the TED dynamic-programming matrix:
  - "update": change the label and/or node_type of the node at source_path,
               and re-parent it if its target_path places it under a different
               parent than its current source_path parent.
  - "delete": remove the subtree rooted at source_path from T1
  - "insert": insert a subtree at the given parent_path / position

Application order:
  1. In-place UPDATE operations (same parent in source and target).
     These only relabel the node; no structural change occurs.

  2. DELETE operations, in reverse preorder (deepest / rightmost paths first).
     Deletes run before re-parenting because their source paths are still
     valid at this stage. Running them after re-parenting would shift indices
     and cause path misses.

  3. Re-parenting UPDATE operations, in two sub-steps:
     a. DETACH: all nodes that must move are popped from their current parent,
        in reverse source-path order (deepest/rightmost first) so that an
        earlier pop never shifts the index of a sibling that is also being
        detached.
     b. ATTACH: detached nodes are inserted at their target position, in
        target-path order (shallowest first) so that a parent node is always
        in place before its children arrive.

  4. INSERT operations, in forward preorder (shallowest first) so that
     parent nodes exist before their children are inserted.

Re-parenting rationale:
The DP aligns flat preorder LD-pair sequences, so a node at source_path in T1
may legitimately belong at a different tree location (target_path) in T2.
This happens when the number of children differs between matching subtrees --
tokens that belong inside one attribute in T2 may be aligned to sibling
attribute-level nodes in T1 (and vice-versa). The target_path stored in every
update operation is the verified tree path in T2, so it is used directly as
the re-parenting destination.

Path representation:
Every path is a list of child indices from the root, e.g.:
  []        -> the root node itself
  [0]       -> first child of root
  [0, 2]    -> third child of the first child of root
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .tree_builder import TreeNode
from .ted_diff import (
    tree_from_dict,
    load_tree_by_country,
    compare_trees,
    save_diff_json,
)


# ---------------------------------------------------------------------------
# Low-level tree navigation helpers
# ---------------------------------------------------------------------------

def get_node(root: TreeNode, path: List[int]) -> TreeNode:
    """
    Return the node reached by following *path* from *root*.
    Raises IndexError if the path is invalid.
    """
    node = root
    for step in path:
        if step < 0 or step >= len(node.children):
            raise IndexError(
                f"Path step {step} out of range "
                f"(node '{node.label}' has {len(node.children)} children); "
                f"full path={path}"
            )
        node = node.children[step]
    return node


def _get_parent_and_index(root: TreeNode, path: List[int]) -> Tuple[TreeNode, int]:
    """
    Return (parent_node, child_index) for the node at *path*.
    Raises ValueError if path is empty (root has no parent).
    """
    if not path:
        raise ValueError("Cannot get parent of root node (empty path).")
    parent = get_node(root, path[:-1])
    return parent, path[-1]


# ---------------------------------------------------------------------------
# Node reconstruction from edit-script subtree snapshots
# ---------------------------------------------------------------------------

def node_from_subtree_dict(d: Dict[str, Any]) -> TreeNode:
    """
    Reconstruct a TreeNode (and its whole subtree) from the subtree snapshot
    embedded in an insert or delete operation.
    """
    node = TreeNode(label=d["label"], node_type=d.get("type", "element"))
    for child_dict in d.get("children", []):
        node.add_child(node_from_subtree_dict(child_dict))
    return node


# ---------------------------------------------------------------------------
# Individual operation applicators
# ---------------------------------------------------------------------------

def apply_inplace_update(root: TreeNode, op: Dict[str, Any]) -> None:
    """
    Relabel the node at op["source_path"] without moving it.
    Used for updates where source and target share the same parent.
    """
    try:
        node = get_node(root, op["source_path"])
    except (IndexError, ValueError):
        return
    node.label = op["to_label"]
    node.node_type = op["to_type"]


def detach_node(root: TreeNode, op: Dict[str, Any]) -> Optional[Tuple[Any, TreeNode]]:
    """
    Pop the node at op["source_path"] from its parent, relabel it, and return
    (op, node) so the caller can re-attach it later.

    Returns None if the path is invalid (node already removed or never existed).
    """
    sp = op["source_path"]
    try:
        parent, idx = _get_parent_and_index(root, sp)
    except (IndexError, ValueError):
        return None
    if idx >= len(parent.children):
        return None
    node = parent.children.pop(idx)
    node.label = op["to_label"]
    node.node_type = op["to_type"]
    return (op, node)


def attach_node(root: TreeNode, op: Dict[str, Any], node: TreeNode) -> None:
    """
    Insert *node* at the position given by op["target_path"].
    Silently skips if the target parent does not exist.
    """
    tp = op["target_path"]
    try:
        tgt_parent = get_node(root, tp[:-1])
    except (IndexError, ValueError):
        print(
            f"[patch] WARNING - attach target parent not found "
            f"(target_path={tp}); node '{node.label}' not placed."
        )
        return
    pos = min(tp[-1], len(tgt_parent.children))
    tgt_parent.children.insert(pos, node)


def apply_delete(root: TreeNode, op: Dict[str, Any]) -> None:
    """
    Delete the subtree rooted at op["source_path"].
    Silently skips if the path no longer exists.
    """
    path = op["source_path"]
    if not path:
        return
    try:
        parent, idx = _get_parent_and_index(root, path)
    except (IndexError, ValueError):
        return
    if idx < len(parent.children):
        parent.children.pop(idx)


def apply_insert(root: TreeNode, op: Dict[str, Any]) -> None:
    """
    Insert the new subtree described by op["subtree"] at op["parent_path"] /
    op["position"]. Silently skips if parent_path is None or invalid.
    """
    parent_path = op.get("parent_path")
    if parent_path is None:
        return
    try:
        parent = get_node(root, parent_path)
    except (IndexError, ValueError):
        return
    position = op.get("position", len(parent.children))
    if position is None:
        position = len(parent.children)
    position = min(position, len(parent.children))
    parent.children.insert(position, node_from_subtree_dict(op["subtree"]))


def filter_top_level_inserts(inserts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop insert operations whose ancestor is also being inserted.
    Since each insert carries a full subtree snapshot, inserting both a node
    and one of its descendants would duplicate content.
    """
    insert_paths = {tuple(op.get("target_path") or []) for op in inserts}
    filtered = []
    for op in inserts:
        path = tuple(op.get("target_path") or [])
        has_inserted_ancestor = any(
            path[:k] in insert_paths for k in range(1, len(path))
        )
        if not has_inserted_ancestor:
            filtered.append(op)
    return filtered


# ---------------------------------------------------------------------------
# Main patching function
# ---------------------------------------------------------------------------

def apply_edit_script(
    source_tree: TreeNode, edit_script: List[Dict[str, Any]]
) -> TreeNode:
    """
    Apply *edit_script* to a deep copy of *source_tree* and return the result.
    The source tree is never modified in place.

    Operation order (see module docstring for rationale):
      1. In-place updates     -- relabel only, no index shifts
      2. Deletes              -- reverse preorder; paths still valid here
      3. Re-parent detach     -- pop movers in reverse source order
      4. Re-parent attach     -- insert movers in forward target order
      5. Inserts              -- forward preorder
    """
    patched = copy.deepcopy(source_tree)

    all_updates = [op for op in edit_script if op["op"] == "update"]
    all_deletes = [op for op in edit_script if op["op"] == "delete"]
    all_inserts = filter_top_level_inserts(
        [op for op in edit_script if op["op"] == "insert"]
    )

    # Classify updates: in-place vs re-parent
    inplace_updates = []
    reparent_updates = []
    for op in all_updates:
        sp = op["source_path"]
        tp = op.get("target_path")
        src_parent = sp[:-1] if sp else None
        tgt_parent = tp[:-1] if tp else None
        if tp is not None and sp and tp and src_parent != tgt_parent:
            reparent_updates.append(op)
        else:
            inplace_updates.append(op)

    # 1. In-place updates
    for op in inplace_updates:
        apply_inplace_update(patched, op)

    # 2. Deletes -- reverse preorder so deeper paths go first
    deletes_sorted = sorted(
        all_deletes,
        key=lambda op: (len(op["source_path"]), op["source_path"]),
        reverse=True,
    )
    for op in deletes_sorted:
        apply_delete(patched, op)

    # 3. Re-parent detach -- reverse source-path order (deepest/rightmost first)
    #    so popping one node does not shift the index of another node that
    #    also needs to be detached at the same level.
    reparent_detach_order = sorted(
        reparent_updates,
        key=lambda op: (len(op["source_path"]), op["source_path"]),
        reverse=True,
    )
    detached: List[Tuple[Dict[str, Any], TreeNode]] = []
    for op in reparent_detach_order:
        result = detach_node(patched, op)
        if result is not None:
            detached.append(result)

    # 4. Re-parent attach -- forward target-path order (shallowest first)
    #    so parent nodes arrive before their children.
    detached_attach_order = sorted(
        detached,
        key=lambda pair: (len(pair[0]["target_path"]), pair[0]["target_path"]),
    )
    for op, node in detached_attach_order:
        attach_node(patched, op, node)

    # 5. Inserts -- forward preorder (shallowest first)
    inserts_sorted = sorted(
        all_inserts,
        key=lambda op: (
            len(op.get("target_path") or []),
            op.get("target_path") or [],
        ),
    )
    for op in inserts_sorted:
        apply_insert(patched, op)

    return patched


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def patch_countries(
    json_path: str,
    country_a: str,
    country_b: str,
) -> Tuple[TreeNode, Dict[str, Any]]:
    """
    Load T1 and T2 from country_trees.json, compute ES(T1, T2), apply it to
    T1, and return (patched_tree, diff_result).
    """
    tree_a = load_tree_by_country(json_path, country_a)
    tree_b = load_tree_by_country(json_path, country_b)

    diff_result = compare_trees(tree_a, tree_b)
    patched = apply_edit_script(tree_a, diff_result["edit_script"])
    return patched, diff_result


def tree_to_dict_full(node: TreeNode) -> Dict[str, Any]:
    """
    Recursively convert a TreeNode to a plain dict (same format as
    TreeNode.to_dict()).  Useful for saving the patched tree to JSON.
    """
    result = {"label": node.label, "type": node.node_type}
    if node.children:
        result["children"] = [tree_to_dict_full(c) for c in node.children]
    return result


def save_patched_tree(patched: TreeNode, output_path: str) -> None:
    """Serialise the patched tree to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree_to_dict_full(patched), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def verify_patch(
    patched: TreeNode,
    target: TreeNode,
) -> Dict[str, Any]:
    """
    Compare the patched tree against the true target tree.

    Returns a dict with:
      - "patched_nodes"       : node count of the patched tree
      - "target_nodes"        : node count of the target tree
      - "residual_ted"        : TED between patched and target (ideally 0)
      - "residual_similarity" : normalised similarity (ideally 1.0)
    """
    residual = compare_trees(patched, target)
    return {
        "patched_nodes": residual["source_node_count"],
        "target_nodes": residual["target_node_count"],
        "residual_ted": residual["ted"],
        "residual_similarity": residual["similarity"],
    }