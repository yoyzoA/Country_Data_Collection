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
    Useful for storing insert operations later.
    """
    result = {
        "label": node.label,
        "type": node.node_type
    }
    if node.children:
        result["children"] = [subtree_to_dict(child) for child in node.children]
    return result


def flatten_tree_preorder(root: TreeNode) -> List[Dict[str, Any]]:
    """
    Chawathe-style flattening:
    preorder traversal, recording a sequence of LD-like entries.

    We keep more than (label, depth) because:
    - node_type helps us distinguish structure vs content
    - path / parent_path / position will help later for patching
    - subtree snapshot helps later for insert operations
    """
    seq: List[Dict[str, Any]] = []

    def visit(node: TreeNode, depth: int, path: Tuple[int, ...], parent_path: Tuple[int, ...] | None, position: int | None):
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
    Cost = 0 if nodes are considered identical, else 1.

    We compare:
    - label
    - node type
    - depth

    Why depth too?
    Because Chawathe’s LD-pair uses (label, depth), not just the label.
    """
    same_label = a["label"] == b["label"]
    same_type = a["node_type"] == b["node_type"]
    same_depth = a["depth"] == b["depth"]

    return 0 if (same_label and same_type and same_depth) else 1


# 4) Dynamic-programming TED over preorder LD-pair sequences
def compute_ted(seq_a: List[Dict[str, Any]], seq_b: List[Dict[str, Any]]) -> Tuple[List[List[int]], int]:
    """
    Compute edit distance between two preorder LD-pair sequences.
    Returns:
        - full DP matrix
        - final TED value
    """
    n = len(seq_a)
    m = len(seq_b)

    dist = [[0] * (m + 1) for _ in range(n + 1)]

    # Base cases
    for i in range(1, n + 1):
        dist[i][0] = dist[i - 1][0] + delete_cost(seq_a[i - 1])

    for j in range(1, m + 1):
        dist[0][j] = dist[0][j - 1] + insert_cost(seq_b[j - 1])

    # Fill matrix
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
    dist: List[List[int]]
) -> List[Dict[str, Any]]:
    """
    Recover one minimum-cost edit script by backtracking the DP matrix.
    """
    i = len(seq_a)
    j = len(seq_b)
    raw_ops: List[Dict[str, Any]] = []

    while i > 0 or j > 0:
        # Case 1: diagonal move (match or update)
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

        # Case 2: deletion
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

        # Case 3: insertion
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

    # Remove costless matches from final human-readable diff
    final_ops = [op for op in raw_ops if op["op"] != "match"]
    return final_ops



# 6) Similarity computation + high-level wrapper
def similarity_from_ted(ted: int, len_a: int, len_b: int) -> float:
    """
    Simple normalized similarity in [0, 1]:
        sim = 1 - ted / (len_a + len_b)

    If both trees are empty, similarity = 1.
    """
    denom = len_a + len_b
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - (ted / denom))


def compare_trees(tree_a: TreeNode, tree_b: TreeNode) -> Dict[str, Any]:
    """
    Full 4.3 pipeline:
    - flatten both trees
    - compute TED
    - backtrack edit script
    - compute similarity
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