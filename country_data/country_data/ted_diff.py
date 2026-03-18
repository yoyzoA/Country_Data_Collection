import json
from typing import Any, Dict, List, Tuple

from .tree_builder import TreeNode


# 1) Rebuild TreeNode objects from the JSON tree dictionaries
def tree_from_dict(data: Dict[str, Any]) -> TreeNode:
    """
    Reconstruct a TreeNode from the dictionary format stored in country_trees.json.
    """
    node = TreeNode(label=data["label"], node_type=data.get("type", "root"))
    for child_data in data.get("children", []):
        node.add_child(tree_from_dict(child_data))
    return node


def load_tree_by_country(json_path: str, country_name: str) -> TreeNode:
    """
    Load one country's tree from country_trees.json and rebuild it as a TreeNode.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if entry["name"].lower() == country_name.lower():
            return tree_from_dict(entry["tree"])

    raise ValueError(f"Country '{country_name}' not found in {json_path}")


# 2) Convert tree to preorder LD-pair sequence
def subtree_to_dict(node: TreeNode) -> Dict[str, Any]:
    """
    Convert a TreeNode subtree back into a plain dictionary.
    Used to embed full subtree snapshots in insert/delete operations,
    which is what the patcher needs to reconstruct or remove nodes.
    """
    result = {
        "label": node.label,
        "type": node.node_type,
    }
    if node.children:
        result["children"] = [subtree_to_dict(child) for child in node.children]
    return result


def flatten_tree_preorder(root: TreeNode) -> List[Dict[str, Any]]:
    """
    Chawathe-style flattening: preorder traversal, producing a sequence of
    LD-like entries.

    Each entry contains:
      - label      : node label (key name, token value, etc.)
      - node_type  : "root" | "element" | "attribute" | "token"
      - depth      : tree depth (used only for the LD-pair, not matching)
      - path       : tuple-path from root (used for patcher addressing)
      - parent_path: path of the parent node (used for insert ops in the patcher)
      - position   : child index under the parent (used for insert ops)
      - subtree    : full subtree dict snapshot (used for insert/delete ops)
    """
    seq: List[Dict[str, Any]] = []

    def visit(
        node: TreeNode,
        depth: int,
        path: Tuple[int, ...],
        parent_path: Tuple[int, ...] | None,
        position: int | None,
    ):
        seq.append({
            "label": node.label,
            "node_type": node.node_type,
            "depth": depth,
            "path": list(path),
            "parent_path": list(parent_path) if parent_path is not None else None,
            "position": position,
            "subtree": subtree_to_dict(node),
        })
        for i, child in enumerate(node.children):
            visit(child, depth + 1, path + (i,), path, i)

    visit(root, depth=0, path=(), parent_path=None, position=None)
    return seq


# 3) Edit operation costs
def insert_cost(_: Dict[str, Any]) -> int:
    return 1


def delete_cost(_: Dict[str, Any]) -> int:
    return 1


def update_cost(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    """
    Cost = 0 if nodes match (same label AND same node type), otherwise 1.

    Design choice:
      We compare only label and node_type — NOT depth.

      Depth is encoded in the LD-pair sequence to reflect structural position,
      but it is NOT a property of the node itself.  Including depth in the
      matching predicate would penalise semantically identical nodes that simply
      appear at a different level after a structural change, inflating the TED
      and producing spurious update operations in the edit script.
      This follows the spirit of Chawathe’s sequence-based comparison, while relaxing 
      depth as a hard matching constraint. 
    """
    return 0 if (a["label"] == b["label"] and a["node_type"] == b["node_type"]) else 1


# 4) Dynamic-programming TED over preorder LD-pair sequences
def compute_ted(
    seq_a: List[Dict[str, Any]],
    seq_b: List[Dict[str, Any]],
) -> Tuple[List[List[int]], int]:
    """
    Compute edit distance between two preorder LD-pair sequences using the
    standard sequence-alignment DP over preorder node sequences, 
    following the spirit of Chawathe’s approach

    Returns:
        dist  – full (n+1) × (m+1) DP matrix (needed for backtracking)
        ted   – the final tree edit distance value
    """
    n, m = len(seq_a), len(seq_b)
    dist = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dist[i][0] = dist[i - 1][0] + delete_cost(seq_a[i - 1])
    for j in range(1, m + 1):
        dist[0][j] = dist[0][j - 1] + insert_cost(seq_b[j - 1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_del = dist[i - 1][j] + delete_cost(seq_a[i - 1])
            cost_ins = dist[i][j - 1] + insert_cost(seq_b[j - 1])
            cost_upd = dist[i - 1][j - 1] + update_cost(seq_a[i - 1], seq_b[j - 1])
            dist[i][j] = min(cost_del, cost_ins, cost_upd)

    return dist, dist[n][m]


# 5) Backtracking to recover the edit script
def backtrack_edit_script(
    seq_a: List[Dict[str, Any]],
    seq_b: List[Dict[str, Any]],
    dist: List[List[int]],
) -> List[Dict[str, Any]]:
    """
    Recover one minimum-cost edit script by backtracking the DP matrix.

    Operations produced:
      - "match"  : node in T1 aligns with an identical node in T2 (cost 0)
      - "update" : node label or type differs between T1 and T2 (cost 1)
      - "delete" : node in T1 has no counterpart in T2 (cost 1)
      - "insert" : node in T2 has no counterpart in T1 (cost 1)

    "match" operations are stripped from the final output (they add no
    information) but are computed internally so the patcher can later
    reconstruct node-address mappings if needed.
    """
    i, j = len(seq_a), len(seq_b)
    raw_ops: List[Dict[str, Any]] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            upd_c = update_cost(seq_a[i - 1], seq_b[j - 1])
            if dist[i][j] == dist[i - 1][j - 1] + upd_c:
                if upd_c == 0:
                    raw_ops.append({
                        "op": "match",
                        "source_path": seq_a[i - 1]["path"],
                        "target_path": seq_b[j - 1]["path"],
                        "label": seq_a[i - 1]["label"],
                        "node_type": seq_a[i - 1]["node_type"],
                    })
                else:
                    raw_ops.append({
                        "op": "update",
                        "source_path": seq_a[i - 1]["path"],
                        "target_path": seq_b[j - 1]["path"],
                        "from_label": seq_a[i - 1]["label"],
                        "to_label": seq_b[j - 1]["label"],
                        "from_type": seq_a[i - 1]["node_type"],
                        "to_type": seq_b[j - 1]["node_type"],
                        "from_depth": seq_a[i - 1]["depth"],
                        "to_depth": seq_b[j - 1]["depth"],
                    })
                i -= 1
                j -= 1
                continue

        # ── delete from T1 ─────────────────────────────────────────────────
        if i > 0 and dist[i][j] == dist[i - 1][j] + delete_cost(seq_a[i - 1]):
            raw_ops.append({
                "op": "delete",
                "source_path": seq_a[i - 1]["path"],
                "label": seq_a[i - 1]["label"],
                "node_type": seq_a[i - 1]["node_type"],
                "depth": seq_a[i - 1]["depth"],
                "subtree": seq_a[i - 1]["subtree"],
            })
            i -= 1
            continue

        # ── insert into T2 ─────────────────────────────────────────────────
        if j > 0 and dist[i][j] == dist[i][j - 1] + insert_cost(seq_b[j - 1]):
            raw_ops.append({
                "op": "insert",
                "target_path": seq_b[j - 1]["path"],
                "parent_path": seq_b[j - 1]["parent_path"],
                "position": seq_b[j - 1]["position"],
                "label": seq_b[j - 1]["label"],
                "node_type": seq_b[j - 1]["node_type"],
                "depth": seq_b[j - 1]["depth"],
                "subtree": seq_b[j - 1]["subtree"],
            })
            j -= 1
            continue

        raise RuntimeError("Backtracking failed: no valid predecessor found.")

    raw_ops.reverse()
    # Strip costless matches from the human-readable diff output
    return [op for op in raw_ops if op["op"] != "match"]


# 6) Similarity computation + high-level wrappers
def similarity_from_ted(ted: int, len_a: int, len_b: int) -> float:
    """
    Normalised similarity in [0, 1]:
        sim = 1 - TED / (|T1| + |T2|)

    Using (|T1| + |T2|) as the denominator gives a value of 0 when every node
    in both trees is deleted/inserted (worst case), and 1 when both trees are
    identical (TED = 0).  This is the Nierman–Jagadish normalisation.

    If both trees are empty, similarity is defined as 1.
    """
    denom = len_a + len_b
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - (ted / denom))


def compare_trees(tree_a: TreeNode, tree_b: TreeNode) -> Dict[str, Any]:
    """
    Full Section 4.3 pipeline:
      1. Flatten both trees into preorder LD-pair sequences.
      2. Compute TED via the DP table.
      3. Backtrack to produce the minimum-cost edit script.
      4. Compute normalised similarity.
    """
    seq_a = flatten_tree_preorder(tree_a)
    seq_b = flatten_tree_preorder(tree_b)

    dist, ted_value = compute_ted(seq_a, seq_b)
    edit_script = backtrack_edit_script(seq_a, seq_b, dist)
    similarity = similarity_from_ted(ted_value, len(seq_a), len(seq_b))

    return {
        "ted": ted_value,
        "similarity": similarity,
        "source_node_count": len(seq_a),
        "target_node_count": len(seq_b),
        "edit_script_length": len(edit_script),
        "edit_script": edit_script,
    }


def compare_countries(json_path: str, country_a: str, country_b: str) -> Dict[str, Any]:
    """
    Load two country trees from country_trees.json and compare them.
    """
    tree_a = load_tree_by_country(json_path, country_a)
    tree_b = load_tree_by_country(json_path, country_b)

    result = compare_trees(tree_a, tree_b)
    result["source_country"] = country_a
    result["target_country"] = country_b
    return result


def save_diff_json(result: Dict[str, Any], output_path: str) -> None:
    """
    Save comparison result and edit script to a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)