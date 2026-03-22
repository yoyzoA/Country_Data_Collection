import json
import os
import sys

from country_data.ted_diff import tree_from_dict
from country_data.postprocess import (
    tree_to_document_dict,
    save_document_json,
    document_dict_to_xml,
    save_document_xml,
    render_infobox_text,
    save_infobox_text,
)


def main():
    base_dir = os.path.dirname(__file__)

    patched_name = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "patched_Lebanon_to_Switzerland.json"
    )
    patched_path = os.path.join(base_dir, patched_name)

    with open(patched_path, "r", encoding="utf-8") as f:
        patched_tree_dict = json.load(f)

    patched_tree = tree_from_dict(patched_tree_dict)

    # 1. Reconstruct document-style dict
    doc = tree_to_document_dict(patched_tree)

    # 2. Save document JSON
    json_out = os.path.join(base_dir, patched_name.replace(".json", "_document.json"))
    save_document_json(doc, json_out)

    # 3. Save XML
    xml_string = document_dict_to_xml(doc, root_tag="country")
    xml_out = os.path.join(base_dir, patched_name.replace(".json", "_document.xml"))
    save_document_xml(xml_string, xml_out)

    # 4. Save infobox-like text
    infobox_text = render_infobox_text(doc)
    txt_out = os.path.join(base_dir, patched_name.replace(".json", "_infobox.txt"))
    save_infobox_text(infobox_text, txt_out)

    print("=" * 65)
    print(" Section 4.5 – Post-processing")
    print("=" * 65)
    print(f"Input patched tree : {patched_path}")
    print(f"Document JSON      : {json_out}")
    print(f"Document XML       : {xml_out}")
    print(f"Infobox text       : {txt_out}")

    print("\n[preview] First lines of infobox-like rendering:\n")
    preview_lines = infobox_text.splitlines()[:15]
    for line in preview_lines:
        print(line)


if __name__ == "__main__":
    main()