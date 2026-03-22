"""
Usage (from inside country_data/):
    python run_postprocess.py [country_name]
    python run_postprocess.py --patched <country_a> <country_b>

Defaults to Lebanon.

What it does  [Section 4.5 – Document Tree Post-Processing]:
1. Loads a document tree from country_trees.json
   OR a previously patched tree from patched_<A>_to_<B>.json.
2. Reconstructs the native JSON document from the tree.
3. Renders the Wikipedia infobox textual format from the tree.
4. Saves both outputs to files in the same directory.
"""

import json
import os
import sys

from country_data.ted_diff import load_tree_by_country, tree_from_dict
from country_data.post_process import (
    tree_to_json,
    tree_to_infobox_text,
    save_json,
    save_infobox_text,
)


def load_patched_tree(base_dir: str, country_a: str, country_b: str):
    """
    Load a previously saved patched tree from patched_<A>_to_<B>.json.
    Returns (TreeNode, file_path).
    """
    safe_a = country_a.replace(" ", "_")
    safe_b = country_b.replace(" ", "_")
    path = os.path.join(base_dir, f"patched_{safe_a}_to_{safe_b}.json")

    if not os.path.exists(path):
        print(f"ERROR: Patched tree file not found: {path}")
        print("Run run_patch.py first to generate it.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return tree_from_dict(data), path


def main():
    base_dir = os.path.dirname(__file__)

    # ── Argument parsing ──────────────────────────────────────────────────────
    use_patched = "--patched" in sys.argv

    if use_patched:
        idx = sys.argv.index("--patched")
        country_a = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "Lebanon"
        country_b = sys.argv[idx + 2] if idx + 2 < len(sys.argv) else "Switzerland"
        tree, source_path = load_patched_tree(base_dir, country_a, country_b)
        label = f"patched_{country_a.replace(' ', '_')}_to_{country_b.replace(' ', '_')}"
        title = f"Patched tree: {country_a} -> {country_b}"
    else:
        country = sys.argv[1] if len(sys.argv) > 1 else "Lebanon"
        json_path = os.path.join(base_dir, "country_trees.json")
        tree = load_tree_by_country(json_path, country)
        label = country.replace(" ", "_")
        title = f"Original tree: {country}"

    print("=" * 65)
    print(f" Section 4.5 - Document Tree Post-Processing")
    print(f" {title}")
    print("=" * 65)

    # ── Step 1: Reconstruct native JSON ──────────────────────────────────────
    print("\n[4.5] Reconstructing native JSON document ...")
    doc = tree_to_json(tree)
    json_out = os.path.join(base_dir, f"postprocessed_{label}.json")
    save_json(doc, json_out)
    print(f"  Fields reconstructed : {len(doc)}")
    print(f"  Saved to             : {json_out}")

    # ── Step 2: Render Wikipedia infobox text ─────────────────────────────────
    print("\n[4.5] Rendering Wikipedia infobox text ...")
    infobox_text = tree_to_infobox_text(tree)
    txt_out = os.path.join(base_dir, f"postprocessed_{label}.txt")
    save_infobox_text(infobox_text, txt_out)
    print(f"  Saved to             : {txt_out}")

    # ── Preview ───────────────────────────────────────────────────────────────
    print(f"\n[preview] Infobox text (first 25 lines):\n")
    lines = infobox_text.split("\n")
    for line in lines[:25]:
        print(f"  {line}")
    if len(lines) > 25:
        print(f"  ... ({len(lines) - 25} more lines)")


if __name__ == "__main__":
    main()
