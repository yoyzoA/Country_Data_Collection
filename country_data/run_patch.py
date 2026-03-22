"""
Usage (from inside country_data/):
    python run_patch.py [country_a] [country_b]

Defaults to Lebanon → Switzerland.

What it does
1. Loads T1 (country_a) and T2 (country_b) from country_trees.json.
2. Calls compare_trees() to compute ES(T1, T2)  [Section 4.3].
3. Applies the edit script to T1 via apply_edit_script()  [Section 4.4].
4. Computes the residual TED between the patched tree and T2 to show how
   well the patch reproduces T2.
5. Saves the patched tree to patched_<A>_to_<B>.json.
6. Saves the diff (edit script) to diff_<A>_to_<B>.json.
"""

import os
import sys
import json

from country_data.ted_diff import load_tree_by_country, compare_trees, save_diff_json
from country_data.patch import (
    apply_edit_script,
    save_patched_tree,
    verify_patch,
    tree_to_dict_full,
)


def main():
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "country_trees.json")

    # Allow overriding from command line
    country_a = sys.argv[1] if len(sys.argv) > 1 else "Lebanon"
    country_b = sys.argv[2] if len(sys.argv) > 2 else "Switzerland"

    print("=" * 65)
    print(f" Section 4.4 - Tree Patching: {country_a} -> {country_b}")
    print("=" * 65)

    # ── Load trees ──────────────────────────────────────────────────────────
    tree_a = load_tree_by_country(json_path, country_a)
    tree_b = load_tree_by_country(json_path, country_b)

    # ── Section 4.3: compute diff ───────────────────────────────────────────
    diff_result = compare_trees(tree_a, tree_b)
    diff_result["source_country"] = country_a
    diff_result["target_country"] = country_b

    print(f"\n[4.3] TED computation")
    print(f"  Source nodes       : {diff_result['source_node_count']}")
    print(f"  Target nodes       : {diff_result['target_node_count']}")
    print(f"  Tree Edit Distance : {diff_result['ted']}")
    print(f"  Similarity         : {diff_result['similarity']:.4f}")
    print(f"  Edit script ops    : {diff_result['edit_script_length']}")

    # Show operation breakdown
    ops = diff_result["edit_script"]
    op_counts = {}
    for op in ops:
        op_counts[op["op"]] = op_counts.get(op["op"], 0) + 1
    print(f"  Operation breakdown: {op_counts}")

    # ── Section 4.4: patch T1 with ES(T1, T2) ──────────────────────────────
    print(f"\n[4.4] Applying edit script to transform {country_a} -> {country_b} ...")
    patched = apply_edit_script(tree_a, ops)

    # ── Verification ────────────────────────────────────────────────────────
    verification = verify_patch(patched, tree_b)
    print(f"\n[4.4] Patch verification")
    print(f"  Patched tree nodes : {verification['patched_nodes']}")
    print(f"  Target tree nodes  : {verification['target_nodes']}")
    print(f"  Residual TED       : {verification['residual_ted']}")
    print(f"  Residual similarity: {verification['residual_similarity']:.4f}")

    if verification["residual_ted"] == 0:
        print("\n  [OK] Patched tree is structurally identical to the target tree!")
    else:
        print(
            f"\n  [WARN] Residual TED = {verification['residual_ted']}. "
            "Some operations could not be applied perfectly due to\n"
            "  LD-sequence approximation (the DP aligns flat sequences, "
            "not true ordered trees)."
        )

    # ── Save outputs ────────────────────────────────────────────────────────
    safe_a = country_a.replace(" ", "_")
    safe_b = country_b.replace(" ", "_")

    diff_path = os.path.join(base_dir, f"diff_{safe_a}_to_{safe_b}.json")
    save_diff_json(diff_result, diff_path)
    print(f"\n[output] Diff saved to    : {diff_path}")

    patch_path = os.path.join(base_dir, f"patched_{safe_a}_to_{safe_b}.json")
    save_patched_tree(patched, patch_path)
    print(f"[output] Patched tree saved: {patch_path}")

    # Preview first 5 edit operations
    print(f"\n[preview] First 5 edit operations:\n")
    for i, op in enumerate(ops[:5], 1):
        print(f"  {i}. {json.dumps(op, ensure_ascii=False)}")


if __name__ == "__main__":
    main()