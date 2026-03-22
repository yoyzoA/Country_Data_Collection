"""
Document Tree Post-Processing  (Section 4.5)

Takes a document tree (TreeNode) produced by tree_builder, ted_diff, or
patch and reconstructs two output representations:

  1. Native JSON document  — the exact inverse of the pre-processing step
     (Section 4.2 / tree_builder.build_tree).  Suitable for storage, further
     comparison, or downstream processing.

  2. Wikipedia infobox textual format  — a human-readable key : value table
     that mirrors the layout of the original Wikipedia infobox.

Reconstruction logic  (inverse of tree_builder.build_tree):
  - Root node  (node_type="root")
      → {"name": root.label, <children …>}
  - Attribute child  (node_type="attribute")  with token grandchildren
      → key = child.label
         value = token labels joined by a single space
  - Element child  (node_type="element")
      → key = child.label
         value = {recursively reconstructed sub-dict}
  - Orphan token child  (node_type="token")  directly under root or an element
      → collected and stored under "_text" (defensive; not produced by the
         normal tree builder but may appear in imperfectly patched trees)

Note on URL field:
  The original scraper stores a "url" field that is intentionally excluded
  from the tree by tree_builder (line: `if key in ("name", "url"): continue`).
  Post-processing therefore cannot reconstruct the URL — this is by design.
"""

import json
from typing import Any, Dict

from .tree_builder import TreeNode


# ── 1. Tree → native JSON dict ───────────────────────────────────────────────

def _element_to_dict(node: TreeNode) -> Dict[str, Any]:
    """
    Recursively reconstruct a plain dict from an element/attribute TreeNode.

    Reads:  node.node_type, node.label, node.children (recursively)
    Returns: dict representing the subtree rooted at *node*
    """
    result: Dict[str, Any] = {}
    orphan_tokens = []

    for child in node.children:
        if child.node_type == "attribute":
            # Re-join tokenised leaf values into a single string.
            result[child.label] = " ".join(t.label for t in child.children)
        elif child.node_type == "element":
            result[child.label] = _element_to_dict(child)
        elif child.node_type == "token":
            # Defensive: token directly under an element node.
            orphan_tokens.append(child.label)

    if orphan_tokens:
        result["_text"] = " ".join(orphan_tokens)

    return result


def tree_to_json(root: TreeNode) -> Dict[str, Any]:
    """
    Reconstruct the original scraped JSON document from a TreeNode tree.

    This is the exact inverse of tree_builder.build_tree():
      - Root label            → "name" field
      - Attribute children    → flat  key: "value string"  entries
      - Element children      → nested key: {sub-dict}  entries

    Reads:  root.label, root.node_type, root.children (recursively)
    Returns: dict ready for json.dump() or further processing

    The reconstructed dict will not contain the "url" field (excluded during
    pre-processing) or any fields that were absent from the original infobox.
    """
    doc: Dict[str, Any] = {"name": root.label}
    orphan_tokens = []

    for child in root.children:
        if child.node_type == "attribute":
            doc[child.label] = " ".join(t.label for t in child.children)
        elif child.node_type == "element":
            doc[child.label] = _element_to_dict(child)
        elif child.node_type == "token":
            orphan_tokens.append(child.label)

    if orphan_tokens:
        doc["_text"] = " ".join(orphan_tokens)

    return doc


# ── 2. Tree → Wikipedia infobox text ─────────────────────────────────────────

def _render_value(key: str, value: Any, indent: int = 0) -> list:
    """
    Recursively render one key/value pair as a list of infobox text lines.

    Flat values produce a single  "  • key  : value"  line.
    Dict values produce a section heading followed by indented sub-entries.
    """
    pad = "  " * indent
    lines = []

    if isinstance(value, dict):
        lines.append(f"{pad}{key}:")
        for sub_key, sub_val in value.items():
            lines.extend(_render_value(sub_key, sub_val, indent + 1))
    else:
        lines.append(f"{pad}  * {key:<22}: {value}")

    return lines


def tree_to_infobox_text(root: TreeNode) -> str:
    """
    Render a TreeNode as a Wikipedia infobox-style plain-text table.

    Reads:   root (TreeNode) — calls tree_to_json internally
    Returns: multi-line string in the following format:

        === Country Name ===

          • Attribute Key        : value
        Section Heading:
            • Sub-key            : sub-value
        ...

    Suitable for display, saving to .txt, or inclusion in a report.
    """
    doc = tree_to_json(root)
    lines = [f"=== {doc.get('name', root.label)} ===", ""]

    for key, value in doc.items():
        if key in ("name", "_text"):
            continue
        lines.extend(_render_value(key, value, indent=0))

    return "\n".join(lines)


# ── 3. Save helpers ───────────────────────────────────────────────────────────

def save_json(doc: Dict[str, Any], output_path: str) -> None:
    """
    Serialise a reconstructed JSON document to a file.

    Reads:  doc         — plain dict produced by tree_to_json()
            output_path — destination file path
    Writes: UTF-8 JSON with 2-space indentation
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def save_infobox_text(text: str, output_path: str) -> None:
    """
    Write an infobox text rendering to a file.

    Reads:  text        — string produced by tree_to_infobox_text()
            output_path — destination file path
    Writes: UTF-8 plain text
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
