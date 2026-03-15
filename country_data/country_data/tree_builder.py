"""
Tree builder fopr the data we extracted
- Root node = country name
- Attribute nodes (flat key-value) = sorted alphabetically, placed before elements
- Element nodes (nested dicts) = maintain document order, placed after attributes
- Leaf nodes = individual tokens from text values (word-by-word tokenization)

Tokenization is seperate node per token to make comparisons easier and more granular.
"""

import re
import json


class TreeNode:

    def __init__(self, label, node_type="root", children=None):
        self.label = label
        self.node_type = node_type
        self.children = children or []

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

    def node_count(self):
        return 1 + sum(child.node_count() for child in self.children)

    def depth(self):
        if not self.children:
            return 0
        return 1 + max(child.depth() for child in self.children)

    def to_dict(self):
        result = {
            "label": self.label,
            "type": self.node_type,
        }
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result

    def pretty_print(self, indent=0, prefix=""):
        type_tag = f"[{self.node_type}]"
        print(f"{prefix}{type_tag} {self.label}")

        for i, child in enumerate(self.children):
            is_last = i == len(self.children) - 1
            child_prefix = prefix + ("    " if is_last else "│   ")
            connector = "└── " if is_last else "├── "
            child._print_with_connector(indent + 1, prefix + connector, child_prefix)

    def _print_with_connector(self, indent, connector, prefix):
        type_tag = f"[{self.node_type}]"
        print(f"{connector}{type_tag} {self.label}")

        for i, child in enumerate(self.children):
            is_last = i == len(self.children) - 1
            child_connector = prefix + ("└── " if is_last else "├── ")
            child_prefix = prefix + ("    " if is_last else "│   ")
            child._print_with_connector(indent + 1, child_connector, child_prefix)


def tokenize(text):
    tokens = re.split(r"\s+", str(text).strip())
    return [t for t in tokens if t]


def build_tree(country_data):
    """
    Ordering rules:
    - Attribute nodes (str values): sorted alphabetically by key name
    - Element nodes (dict values): maintain original document order
    - Attributes appear BEFORE elements at each level
    """
    name = country_data.get("name", "Unknown")
    root = TreeNode(label=name, node_type="root")
    
    attributes = []
    elements = []

    for key, value in country_data.items():
        if key in ("name", "url"):
            continue
        if isinstance(value, dict):
            elements.append((key, value))
        else:
            attributes.append((key, value))

    # Sort attributes alphabetically by key
    attributes.sort(key=lambda x: x[0].lower())

    # Add attribute nodes first (sorted A-Z)
    for key, value in attributes:
        attr_node = TreeNode(label=key, node_type="attribute")

        # Tokenize the value and add each token as a leaf child
        tokens = tokenize(value)
        for token in tokens:
            token_node = TreeNode(label=token, node_type="token")
            attr_node.add_child(token_node)

        root.add_child(attr_node)

    # Add element nodes after (document order preserved)
    for key, value in elements:
        elem_node = _build_element_node(key, value)
        root.add_child(elem_node)

    return root


def _build_element_node(key, value_dict):
    """
    Recursively build an element node from a nested dictionary.

    Applies the same ordering rules at each level:
    - Attributes (str values) sorted alphabetically, placed first
    - Elements (dict values) in document order, placed after
    """
    elem_node = TreeNode(label=key, node_type="element")

    sub_attrs = []
    sub_elems = []

    for k, v in value_dict.items():
        if isinstance(v, dict):
            sub_elems.append((k, v))
        else:
            sub_attrs.append((k, v))

    # Sort sub-attributes alphabetically
    sub_attrs.sort(key=lambda x: x[0].lower())

    # Add sub-attributes first
    for k, v in sub_attrs:
        attr_node = TreeNode(label=k, node_type="attribute")
        tokens = tokenize(v)
        for token in tokens:
            token_node = TreeNode(label=token, node_type="token")
            attr_node.add_child(token_node)
        elem_node.add_child(attr_node)

    # Add sub-elements after (recursively)
    for k, v in sub_elems:
        child_elem = _build_element_node(k, v)
        elem_node.add_child(child_elem)

    return elem_node


def build_all_trees(countries_data):
    """
    Build trees for a list of country dictionaries.

    Args:
        countries_data: List of country dicts from scraped JSON.

    Returns:
        Dict mapping country name to its TreeNode root.
    """
    trees = {}
    for country in countries_data:
        name = country.get("name", "Unknown")
        trees[name] = build_tree(country)
    return trees


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tree_builder.py countries.json [country_name]")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    # If a country name is given, only show that tree
    if len(sys.argv) >= 3:
        country_name = " ".join(sys.argv[2:])
        matches = [d for d in data if d["name"].lower() == country_name.lower()]
        if not matches:
            print(f"Country '{country_name}' not found.")
            sys.exit(1)
        tree = build_tree(matches[0])
        tree.pretty_print()
        print(f"\nTotal nodes: {tree.node_count()}")
        print(f"Max depth: {tree.depth()}")
    else:
        # Show stats for all countries
        trees = build_all_trees(data)
        print(f"Built trees for {len(trees)} countries.\n")
        for name, tree in list(trees.items())[:5]:
            print(f"{name}: {tree.node_count()} nodes, depth {tree.depth()}")
        if len(trees) > 5:
            print(f"... and {len(trees) - 5} more.")
