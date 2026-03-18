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
  - "update": change the label and/or node_type of the node at source_path
  - "delete": remove the subtree rooted at source_path from T1
  - "insert": insert a subtree at the given parent_path / position

Since updates do not change tree shape, they are applied first.
Deletes are then applied in reverse preorder (deepest / latest paths first)
so that removing a node does not invalidate the paths of nodes that still
need to be deleted.
Inserts are applied afterwards in forward preorder (shallowest first) so that
parent locations are available before descendant insertions are considered.

Note:
The edit script is derived from a flattened preorder alignment rather than
from an exact subtree-aware tree mapping. For that reason, patching is an
approximation step, and the patched tree is checked afterwards by comparing
it to the true target tree.

Path representation:
Every path is a list of child indices from the root, e.g.:
  []        -> the root node itself
  [0]       -> first child of root
  [0, 2]    -> third child of the first child of root

Helpers:
  - get_node()
  - _get_parent_and_index()
  - node_from_subtree_dict()
  - apply_update()
  - apply_delete()
  - apply_insert()
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


# Low-level tree navigation helpers
def get_node(root: TreeNode, path: List[int]) -> TreeNode:
    """
    Return the node reached by following *path* from *root*.
    Raises IndexError / ValueError if the path is invalid.
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
    The path must have at least one element (you cannot ask for the parent of
    the root).
    """
    if not path:
        raise ValueError("Cannot get parent of root node (empty path).")
    parent = get_node(root, path[:-1])
    return parent, path[-1]


# Node reconstruction from edit-script subtree snapshots
def node_from_subtree_dict(d: Dict[str, Any]) -> TreeNode:
    """
    Reconstruct a TreeNode (and its whole subtree) from the subtree snapshot
    embedded in an insert or delete operation.
    """
    node = TreeNode(label=d["label"], node_type=d.get("type", "element"))
    for child_dict in d.get("children", []):
        node.add_child(node_from_subtree_dict(child_dict))
    return node


# Individual operation applicators
def apply_update(root: TreeNode, op: Dict[str, Any]) -> None:
    """
    Update the label and node_type of the node at op["source_path"].
    """
    path = op["source_path"]
    node = get_node(root, path)
    node.label = op["to_label"]
    node.node_type = op["to_type"]


def apply_delete(root: TreeNode, op: Dict[str, Any]) -> None:
    """
    Delete the subtree rooted at op["source_path"].

    Note: deleting a node also deletes all its descendants, but the edit
    script may contain separate delete operations for those descendants.
    We guard against double-deletion by skipping paths that no longer exist.
    """
    path = op["source_path"]
    if not path:
        # Deleting the root is not meaningful in our context; skip silently.
        return
    try:
        parent, idx = _get_parent_and_index(root, path)
    except (IndexError, ValueError):
        # Node was already removed as part of an ancestor's deletion.
        return
    if idx < len(parent.children):
        parent.children.pop(idx)


def apply_insert(root: TreeNode, op: Dict[str, Any]) -> None:
    """
    Insert the new subtree described by op["subtree"] at the position indicated
    by op["parent_path"] and op["position"].

    If the parent_path is None this is an insert at the root level, which we
    treat as a no-op (the root update already handles root-level changes).

    We clamp the insertion index to len(parent.children) so that operations
    generated for the *target* tree's structure never fail even when the
    current tree has fewer children than expected.
    """
    parent_path = op.get("parent_path")
    if parent_path is None:
        return  # Cannot insert above the root; skip.

    try:
        parent = get_node(root, parent_path)
    except (IndexError, ValueError):
        # Parent hasn't been created yet or was removed; skip this op.
        return

    position = op.get("position", len(parent.children))
    if position is None:
        position = len(parent.children)
    # Clamp to valid range
    position = min(position, len(parent.children))

    new_node = node_from_subtree_dict(op["subtree"])
    parent.children.insert(position, new_node)


def filter_top_level_inserts(inserts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only insert operations whose ancestor is NOT also being inserted.

    Since each insert operation stores a full subtree snapshot, inserting both
    a node and one of its descendants would duplicate content.
    """
    insert_paths = {tuple(op.get("target_path") or []) for op in inserts}
    filtered = []

    for op in inserts:
        path = tuple(op.get("target_path") or [])
        has_inserted_ancestor = any(path[:k] in insert_paths for k in range(1, len(path)))
        if not has_inserted_ancestor:
            filtered.append(op)

    return filtered

# Main patching function
def apply_edit_script(source_tree: TreeNode, edit_script: List[Dict[str, Any]]) -> TreeNode:
    """
    Apply *edit_script* to a deep copy of *source_tree* and return the result.

    The source tree is never modified in place.

    Application order:
      1. UPDATE operations  – applied first; paths still valid on original shape.
      2. DELETE operations  – applied in *reverse preorder* (deepest first) so
                             removing a node never invalidates the path of a
                             sibling that must also be deleted.
      3. INSERT operations  – applied in *forward preorder* (shallowest first)
                             so parents exist before children are added.
    """
    # Work on a deep copy so the caller's tree is untouched
    patched = copy.deepcopy(source_tree)

    updates = [op for op in edit_script if op["op"] == "update"]
    deletes = [op for op in edit_script if op["op"] == "delete"]
    inserts = [op for op in edit_script if op["op"] == "insert"]

    inserts = filter_top_level_inserts(inserts)

    # 1. Updates
    for op in updates:
        try:
            apply_update(patched, op)
        except (IndexError, ValueError) as e:
            # Log but continue: a failed update is non-fatal
            print(f"[patch] WARNING – update skipped ({e}): {op}")

    # 2. Deletes (reverse preorder = deepest first) 
    deletes_sorted = sorted(
        deletes,
        key=lambda op: (len(op["source_path"]), op["source_path"]),
        reverse=True,
    )
    for op in deletes_sorted:
        try:
            apply_delete(patched, op)
        except (IndexError, ValueError) as e:
            print(f"[patch] WARNING – delete skipped ({e}): {op}")

    # 3. Inserts (forward preorder = shallowest first)
    # Sort by target_path length ascending so root-level inserts come first.
    inserts_sorted = sorted(
        inserts,
        key=lambda op: (len(op.get("target_path") or []), op.get("target_path") or []),
    )
    for op in inserts_sorted:
        try:
            apply_insert(patched, op)
        except (IndexError, ValueError) as e:
            print(f"[patch] WARNING – insert skipped ({e}): {op}")

    return patched


# High-level helpers
def patch_countries(
    json_path: str,
    country_a: str,
    country_b: str,
) -> Tuple[TreeNode, Dict[str, Any]]:
    """
    Load T1 and T2 from country_trees.json, compute ES(T1, T2), apply it to
    T1, and return (patched_tree, diff_result).

    The returned *patched_tree* should structurally match T2.
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
    """
    Serialise the patched tree to a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree_to_dict_full(patched), f, indent=2, ensure_ascii=False)


# Verification helper
def verify_patch(
    patched: TreeNode,
    target: TreeNode,
) -> Dict[str, Any]:
    """
    Compare the patched tree against the true target tree to measure how well
    the patch reproduced T2.

    Returns a dict with:
      - "patched_nodes"   : node count of the patched tree
      - "target_nodes"    : node count of the target tree
      - "residual_ted"    : TED between patched and target (ideally 0)
      - "residual_similarity" : normalised similarity (ideally 1.0)
    """
    residual = compare_trees(patched, target)
    return {
        "patched_nodes": residual["source_node_count"],
        "target_nodes": residual["target_node_count"],
        "residual_ted": residual["ted"],
        "residual_similarity": residual["similarity"],
    }