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
      - label       : node label (key name, token value, etc.)
      - node_type   : "root" | "element" | "attribute" | "token"
      - depth       : tree depth (used only for the LD-pair, not matching)
      - path        : tuple-path from root (used for patcher addressing)
      - parent_path : path of the parent node (used for insert ops in the patcher)
      - parent_label: label of the parent node (used in update_cost to prevent
                      tokens from being matched across different attribute parents)
      - position    : child index under the parent (used for insert ops)
      - subtree     : full subtree dict snapshot (used for insert/delete ops)
    """
    seq: List[Dict[str, Any]] = []

    def visit(
        node: TreeNode,
        depth: int,
        path: Tuple[int, ...],
        parent_path: Tuple[int, ...] | None,
        position: int | None,
        parent_label: str | None,
    ):
        seq.append({
            "label": node.label,
            "node_type": node.node_type,
            "depth": depth,
            "path": list(path),
            "parent_path": list(parent_path) if parent_path is not None else None,
            "parent_label": parent_label,
            "position": position,
            "subtree": subtree_to_dict(node),
        })
        for i, child in enumerate(node.children):
            visit(child, depth + 1, path + (i,), path, i, parent_label=node.label)

    visit(root, depth=0, path=(), parent_path=None, position=None, parent_label=None)
    return seq


# 3) Edit operation costs
def insert_cost(_: Dict[str, Any]) -> int:
    return 1


def delete_cost(_: Dict[str, Any]) -> int:
    return 1


def update_cost(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    """
    Cost of aligning node a (from T1) with node b (from T2).

      0  – exact match: same label AND same node_type (and, for tokens,
           same parent attribute label)
      1  – same-type relabel: node_type matches, labels differ (and, for
           tokens, same parent attribute label)
      2  – cross-type: node_type differs (attribute vs token, etc.)
           OR same-type token whose parent attribute labels differ

    Cross-type cost is set to 2 so that it equals the cost of a separate
    delete + insert pair (1 + 1 = 2).  The backtracker is ordered to prefer
    delete/insert over a cross-type diagonal step when both have equal cost,
    which eliminates structurally nonsensical matches such as an attribute
    node being aligned against a token node.

    The cross-parent-label cost for tokens prevents token "borrowing" across
    attribute boundaries.  For example, a token under the "Currency" attribute
    in T1 should never be aligned with a token under the "Capital" attribute
    in T2; assigning cost 2 to such a match makes the DP prefer a separate
    delete + insert instead.  This constraint applies to BOTH relabels AND
    exact-label matches: a "(" token under "Area" must not be matched at cost
    0 to a "(" token under "Currency", as that would prevent the Currency "("
    from being emitted as an INSERT.

    Depth is intentionally excluded from the cost (see original note): two
    nodes with the same label and type are considered a valid match regardless
    of their depth in the tree.
    """
    if a["node_type"] != b["node_type"]:
        return 2  # cross-type: never preferred over delete + insert
    # Same node type.
    # For tokens, apply parent-label constraint BEFORE the exact-match check
    # so that cross-attribute tokens are never matched at cost 0.
    if a["node_type"] == "token":
        pa = a.get("parent_label")
        pb = b.get("parent_label")
        if pa is not None and pb is not None and pa != pb:
            return 2  # cross-parent token: prefer delete + insert
    if a["label"] == b["label"]:
        return 0  # exact match
    return 1  # same-type relabel


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
      - "update" : same-type nodes whose label differs (cost 1)
      - "delete" : node in T1 has no counterpart in T2 (cost 1)
      - "insert" : node in T2 has no counterpart in T1 (cost 1)

    Backtracking priority:
      1. Same-type diagonal (match / update, upd_c <= 1) — preferred.
      2. Delete from T1.
      3. Insert from T2.
      4. Cross-type diagonal (upd_c == 2) — last resort only.

    Cross-type updates are pushed to last resort so that the algorithm
    emits separate delete + insert pairs instead.  This is mathematically
    safe: whenever the cross-type diagonal is the DP optimum (cost 2), a
    delete path of equal cost is always available (dist[i-1][j] + 1 <=
    dist[i-1][j-1] + 2 = dist[i][j]), so the fallback block is never
    reached in practice.

    "match" operations are stripped from the final output.
    """
    i, j = len(seq_a), len(seq_b)
    raw_ops: List[Dict[str, Any]] = []

    while i > 0 or j > 0:
        # ── 1. Same-type diagonal (match / same-type update) ───────────────
        if i > 0 and j > 0:
            upd_c = update_cost(seq_a[i - 1], seq_b[j - 1])
            if upd_c <= 1 and dist[i][j] == dist[i - 1][j - 1] + upd_c:
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

        # ── 2. Delete from T1 ──────────────────────────────────────────────
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

        # ── 3. Insert from T2 ──────────────────────────────────────────────
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

        # ── 4. Cross-type diagonal (fallback — should not occur) ───────────
        if i > 0 and j > 0:
            upd_c = update_cost(seq_a[i - 1], seq_b[j - 1])
            if dist[i][j] == dist[i - 1][j - 1] + upd_c:
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

        raise RuntimeError("Backtracking failed: no valid predecessor found.")

    raw_ops.reverse()
    # Keep matches in the output; _fix_doomed_updates will strip them after
    # converting any doomed ones to INSERT operations.
    return raw_ops


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


def _fix_doomed_updates(
    edit_script: List[Dict[str, Any]],
    seq_b: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Post-process the raw edit script to fix two classes of structural problems,
    then strip surviving MATCH operations.

    Pass 1 -- Doomed updates/matches:
        When a parent node is being deleted, all UPDATE and MATCH ops on its
        descendants are "doomed" -- the relabelled/matched nodes vanish with the
        parent deletion.  Convert them to INSERT ops (full T2 subtree) so the
        patcher can place the correct content at the right location.

    Pass 2 -- Covered updates/matches:
        When an ancestor of a MATCH/UPDATE's *target* is being inserted (i.e.,
        an INSERT op already covers the target), the source node will end up in
        the wrong structural location after patching.  The INSERT already
        provides the correct T2 content, so the source node just needs to be
        deleted.  Example: France Area.Total matched to Germany Area.Total, but
        France Area [28] is renamed to Formation [16] and Germany Area [17] is
        inserted separately -- leaving a spurious "Total" inside Formation.
        Converting the match to a DELETE removes that spurious node.
    """
    # Build the set of source paths that are directly deleted.
    deleted_paths = {
        tuple(op["source_path"])
        for op in edit_script
        if op["op"] == "delete"
    }

    # Build a lookup from target path to the seq_b entry (for subtree snapshots).
    seq_b_by_path: Dict[Tuple, Dict[str, Any]] = {
        tuple(n["path"]): n for n in seq_b
    }

    # -- Pass 1: doomed updates / doomed matches ------------------------------
    pass1: List[Dict[str, Any]] = []
    for op in edit_script:
        if op["op"] in ("update", "match"):
            sp = tuple(op["source_path"])
            # A node is "doomed" if any of its proper ancestors is being deleted.
            is_doomed = any(
                sp[:k] in deleted_paths
                for k in range(1, len(sp))
            )
            if is_doomed:
                # Rescue: emit an INSERT at the target position using the full
                # T2 subtree, then drop the (now-redundant) update/match.
                tp = tuple(op["target_path"])
                target_node = seq_b_by_path.get(tp)
                if target_node is not None:
                    pass1.append({
                        "op": "insert",
                        "target_path": list(tp),
                        "parent_path": target_node["parent_path"],
                        "position": target_node["position"],
                        "label": target_node["label"],
                        "node_type": target_node["node_type"],
                        "depth": target_node["depth"],
                        "subtree": target_node["subtree"],
                    })
                continue  # drop the original update/match
        pass1.append(op)

    # -- Pass 2: covered updates / covered matches ----------------------------
    # Collect all INSERT target paths produced so far (original inserts +
    # those promoted from doomed updates/matches in pass 1).
    inserted_target_paths = {
        tuple(op["target_path"])
        for op in pass1
        if op["op"] == "insert"
    }

    pass2: List[Dict[str, Any]] = []
    for op in pass1:
        if op["op"] in ("update", "match"):
            tp = tuple(op["target_path"])
            # A target is "covered" if any of its proper ancestors is being
            # inserted -- the INSERT subtree snapshot already contains the
            # correct T2 content for this node and all its descendants.
            is_covered = any(
                tp[:k] in inserted_target_paths
                for k in range(1, len(tp))
            )
            if is_covered:
                # The INSERT will provide the correct T2 content; delete the
                # source node so it does not linger in the wrong location.
                pass2.append({
                    "op": "delete",
                    "source_path": op["source_path"],
                    "label": op.get("from_label", op.get("label", "")),
                    "node_type": op.get("from_type", op.get("node_type", "")),
                    "depth": op.get("from_depth", op.get("depth", 0)),
                    "subtree": {},
                })
                continue
            if op["op"] == "match":
                continue  # Non-covered, non-doomed match: strip it
        pass2.append(op)

    return pass2


def compare_trees(tree_a: TreeNode, tree_b: TreeNode) -> Dict[str, Any]:
    """
    Full Section 4.3 pipeline:
      1. Flatten both trees into preorder LD-pair sequences.
      2. Compute TED via the DP table.
      3. Backtrack to produce the minimum-cost edit script.
      4. Fix doomed updates (UPDATE ops whose source node will be destroyed
         by an ancestor DELETE — convert them to INSERT ops at the T2 position).
      5. Compute normalised similarity.
    """
    seq_a = flatten_tree_preorder(tree_a)
    seq_b = flatten_tree_preorder(tree_b)

    dist, ted_value = compute_ted(seq_a, seq_b)
    edit_script = backtrack_edit_script(seq_a, seq_b, dist)
    edit_script = _fix_doomed_updates(edit_script, seq_b)
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