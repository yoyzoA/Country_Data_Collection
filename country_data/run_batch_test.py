"""
Batch test runner for the full pipeline (Sections 4.3, 4.4, 4.5).

Runs multiple country-pair patches, collects metrics, and prints a summary.
"""

import os
import sys
import json

from country_data.ted_diff import load_tree_by_country, compare_trees
from country_data.patch import apply_edit_script, verify_patch, tree_to_dict_full
from country_data.post_process import tree_to_json, tree_to_infobox_text, save_json, save_infobox_text


BASE_DIR = os.path.dirname(__file__)
JSON_PATH = os.path.join(BASE_DIR, "country_trees.json")

TEST_PAIRS = [
    ("Lebanon",   "Switzerland"),
    ("France",    "Germany"),
    ("Japan",     "Canada"),
    ("Brazil",    "Australia"),
    ("India",     "China"),
    ("Spain",     "Italy"),
    ("Egypt",     "Turkey"),
]


def run_pair(country_a, country_b):
    """Run the full pipeline for one country pair. Returns a result dict."""
    try:
        tree_a = load_tree_by_country(JSON_PATH, country_a)
        tree_b = load_tree_by_country(JSON_PATH, country_b)
    except ValueError as e:
        return {"error": str(e)}

    # 4.3 – TED
    diff = compare_trees(tree_a, tree_b)

    ops = diff["edit_script"]
    op_counts = {}
    for op in ops:
        op_counts[op["op"]] = op_counts.get(op["op"], 0) + 1

    # Count cross-type updates (should be 0 with the fix)
    cross_type_updates = sum(
        1 for op in ops
        if op["op"] == "update" and op.get("from_type") != op.get("to_type")
    )

    # 4.4 – Patch
    patched = apply_edit_script(tree_a, ops)
    verif = verify_patch(patched, tree_b)

    # 4.5 – Postprocess
    doc = tree_to_json(patched)
    doc_b = tree_to_json(tree_b)

    # Field-level comparison: how many fields match exactly?
    total_fields = len(doc_b)
    matching_fields = sum(1 for k in doc_b if doc_b.get(k) == doc.get(k))
    reconstructed_fields = sum(1 for k in doc_b if k in doc)

    # Save outputs
    safe_a = country_a.replace(" ", "_")
    safe_b = country_b.replace(" ", "_")

    patch_path = os.path.join(BASE_DIR, f"patched_{safe_a}_to_{safe_b}.json")
    with open(patch_path, "w", encoding="utf-8") as f:
        json.dump(tree_to_dict_full(patched), f, indent=2, ensure_ascii=False)

    pp_json_path = os.path.join(BASE_DIR, f"postprocessed_patched_{safe_a}_to_{safe_b}.json")
    save_json(doc, pp_json_path)

    pp_txt_path = os.path.join(BASE_DIR, f"postprocessed_patched_{safe_a}_to_{safe_b}.txt")
    infobox = tree_to_infobox_text(patched)
    save_infobox_text(infobox, pp_txt_path)

    return {
        "country_a":            country_a,
        "country_b":            country_b,
        "src_nodes":            diff["source_node_count"],
        "tgt_nodes":            diff["target_node_count"],
        "ted":                  diff["ted"],
        "similarity":           diff["similarity"],
        "edit_ops":             diff["edit_script_length"],
        "op_counts":            op_counts,
        "cross_type_updates":   cross_type_updates,
        "residual_ted":         verif["residual_ted"],
        "residual_similarity":  verif["residual_similarity"],
        "total_fields":         total_fields,
        "reconstructed_fields": reconstructed_fields,
        "matching_fields":      matching_fields,
        "patch_path":           patch_path,
        "pp_json_path":         pp_json_path,
        "pp_txt_path":          pp_txt_path,
    }


def print_result(r):
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return

    print(f"  Nodes          : {r['src_nodes']} -> {r['tgt_nodes']}")
    print(f"  TED / Similarity: {r['ted']} / {r['similarity']:.4f}")
    print(f"  Edit ops       : {r['edit_ops']}  {r['op_counts']}")
    print(f"  Cross-type upd : {r['cross_type_updates']}")
    print(f"  Residual TED   : {r['residual_ted']}")
    print(f"  Residual sim   : {r['residual_similarity']:.4f}")
    print(f"  Fields: {r['reconstructed_fields']}/{r['total_fields']} reconstructed, "
          f"{r['matching_fields']}/{r['total_fields']} exact matches")


def main():
    print("=" * 70)
    print("  Batch Pipeline Test  (Sections 4.3 + 4.4 + 4.5)")
    print("=" * 70)

    results = []
    for country_a, country_b in TEST_PAIRS:
        header = f" {country_a} -> {country_b} "
        print(f"\n{'-' * 70}")
        print(f"{header:^70}")
        print(f"{'-' * 70}")
        r = run_pair(country_a, country_b)
        print_result(r)
        results.append(r)

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Pair':<35} {'ResidTED':>8} {'Fields':>12} {'XType':>6}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            pair = f"  ERROR"
            print(f"{pair:<35}")
            continue
        pair = f"{r['country_a']} -> {r['country_b']}"
        fields = f"{r['matching_fields']}/{r['total_fields']}"
        print(f"{pair:<35} {r['residual_ted']:>8} {fields:>12} {r['cross_type_updates']:>6}")

    # Check for any cross-type updates (should be 0 after fix)
    all_cross = sum(r.get("cross_type_updates", 0) for r in results if "error" not in r)
    print(f"\nTotal cross-type updates across all pairs: {all_cross}")
    if all_cross == 0:
        print("[OK] No cross-type updates detected. Fix is working correctly.")
    else:
        print("[WARN] Cross-type updates found. Review ted_diff.py fix.")


if __name__ == "__main__":
    main()
