"""
This module converts a patched / compared TreeNode back into:
  1. a document-style nested Python dictionary
  2. JSON
  3. XML
  4. a simple infobox-like text rendering

The goal is to reverse the preprocessing idea from Section 4.2:
    document  -> tree
and now do:
    tree -> document-like representation

Tree interpretation:
  - root      : whole country document
  - element   : nested object / section
  - attribute : key whose value is usually a token string
  - token     : content text
"""

import json
from typing import Any, Dict, List
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from .tree_builder import TreeNode


# Low-level helpers
def collect_tokens(node: TreeNode) -> str:
    """
    Recursively collect all token descendants of a node and join them into one string.

    This is useful when an attribute node contains token children such as:
        Currency -> ["Swiss", "franc"]
    which should become:
        "Swiss franc"
    """
    tokens: List[str] = []

    def visit(n: TreeNode) -> None:
        if n.node_type == "token":
            tokens.append(n.label)
        for child in n.children:
            visit(child)

    visit(node)
    return " ".join(tokens).strip()


def is_token_only_subtree(node: TreeNode) -> bool:
    """
    Return True if all descendants under this node are token nodes only.
    """
    if not node.children:
        return False

    def check(n: TreeNode) -> bool:
        for child in n.children:
            if child.node_type != "token":
                return False
            if child.children:
                return False
        return True

    return check(node)


# Core reconstruction: TreeNode -> document-style dict
def tree_to_document_dict(root: TreeNode) -> Dict[str, Any]:
    """
    Convert a TreeNode tree back into a document-style nested dictionary.

    Design:
      - The root label is stored as "name"
      - Attribute nodes become key-value pairs
      - Element nodes become nested dictionaries
      - Token children are merged back into strings
    """
    if root.node_type != "root":
        raise ValueError("tree_to_document_dict expects the root TreeNode.")

    doc: Dict[str, Any] = {"name": root.label}

    for child in root.children:
        key, value = node_to_key_value(child)
        _merge_key_value(doc, key, value)

    return doc


def node_to_key_value(node: TreeNode) -> tuple[str, Any]:
    """
    Convert one node into a (key, value) pair suitable for a document dict.

    Cases:
      - attribute + token children  -> "Key": "joined token string"
      - attribute + nested content  -> "Key": nested dict or mixed structure
      - element                     -> "Section": {...}
    """
    if node.node_type == "attribute":
        if is_token_only_subtree(node):
            return node.label, collect_tokens(node)

        if not node.children:
            return node.label, ""

        # Mixed/nested attribute content
        nested: Dict[str, Any] = {}
        loose_tokens: List[str] = []

        for child in node.children:
            if child.node_type == "token":
                loose_tokens.append(child.label)
            else:
                k, v = node_to_key_value(child)
                _merge_key_value(nested, k, v)

        if loose_tokens and not nested:
            return node.label, " ".join(loose_tokens).strip()

        if loose_tokens:
            nested["_text"] = " ".join(loose_tokens).strip()

        return node.label, nested

    if node.node_type == "element":
        nested: Dict[str, Any] = {}

        for child in node.children:
            if child.node_type == "token":
                # Rare case: loose token directly under element
                text = nested.get("_text", "")
                nested["_text"] = (text + " " + child.label).strip()
            else:
                k, v = node_to_key_value(child)
                _merge_key_value(nested, k, v)

        return node.label, nested

    if node.node_type == "token":
        # Usually tokens are handled by parents, but keep a fallback
        return node.label, node.label

    # Fallback for unexpected node types
    nested: Dict[str, Any] = {}
    for child in node.children:
        k, v = node_to_key_value(child)
        _merge_key_value(nested, k, v)
    return node.label, nested


def _merge_key_value(target: Dict[str, Any], key: str, value: Any) -> None:
    """
    Merge a key/value into a dictionary.

    If a key appears multiple times, store the values as a list.
    This is useful if reconstructed structures contain repeated keys.
    """
    if key not in target:
        target[key] = value
        return

    existing = target[key]
    if isinstance(existing, list):
        existing.append(value)
    else:
        target[key] = [existing, value]


# JSON output
def save_document_json(doc: Dict[str, Any], output_path: str) -> None:
    """
    Save reconstructed document dictionary as JSON.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


# XML output
def document_dict_to_xml(doc: Dict[str, Any], root_tag: str = "country") -> str:
    """
    Convert a reconstructed document dictionary into a pretty XML string.
    """
    root = Element(root_tag)
    _append_xml_children(root, doc)
    rough = tostring(root, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")


def _append_xml_children(parent: Element, value: Any, key_name: str | None = None) -> None:
    """
    Recursive helper to build XML from nested dict/list/scalar values.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            child = SubElement(parent, _safe_xml_tag(k))
            _append_xml_children(child, v, k)

    elif isinstance(value, list):
        item_tag = "item" if key_name is None else _safe_xml_tag(key_name[:-1] if key_name.endswith("s") else key_name)
        for item in value:
            child = SubElement(parent, item_tag)
            _append_xml_children(child, item, item_tag)

    else:
        parent.text = "" if value is None else str(value)


def _safe_xml_tag(tag: str) -> str:
    """
    Make a simple XML-safe tag name.
    """
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in tag.strip())
    if not cleaned:
        cleaned = "field"
    if cleaned[0].isdigit():
        cleaned = "field_" + cleaned
    return cleaned


def save_document_xml(xml_string: str, output_path: str) -> None:
    """
    Save XML string to file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_string)


# Infobox-like text rendering
def render_infobox_text(doc: Dict[str, Any]) -> str:
    """
    Render the reconstructed document dictionary in a simple infobox-like text form.
    """
    lines: List[str] = []

    name = doc.get("name", "Unknown")
    lines.append(f"Country: {name}")
    lines.append("=" * (9 + len(str(name))))

    for key, value in doc.items():
        if key == "name":
            continue
        _render_field(lines, key, value, indent=0)

    return "\n".join(lines)


def _render_field(lines: List[str], key: str, value: Any, indent: int) -> None:
    prefix = "  " * indent

    if isinstance(value, dict):
        lines.append(f"{prefix}{key}:")
        for k, v in value.items():
            _render_field(lines, k, v, indent + 1)

    elif isinstance(value, list):
        lines.append(f"{prefix}{key}:")
        for item in value:
            if isinstance(item, (dict, list)):
                _render_field(lines, "-", item, indent + 1)
            else:
                lines.append(f"{prefix}  - {item}")

    else:
        lines.append(f"{prefix}{key}: {value}")


def save_infobox_text(text: str, output_path: str) -> None:
    """
    Save infobox-like text rendering to file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)