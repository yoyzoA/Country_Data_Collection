import os
import json

from country_data.ted_diff import compare_countries, save_diff_json


def main():
    json_path = os.path.join(os.path.dirname(__file__), "country_trees.json")

    # CHANGE THESE TWO COUNTRY NAMES TO TEST DIFFERENT PAIRS
    country_a = "Lebanon"
    country_b = "Syria"

    result = compare_countries(json_path, country_a, country_b)

    # Print a compact summary
    print("=" * 60)
    print(f"Source country       : {result['source_country']}")
    print(f"Target country       : {result['target_country']}")
    print(f"Source node count    : {result['source_node_count']}")
    print(f"Target node count    : {result['target_node_count']}")
    print(f"Tree Edit Distance   : {result['ted']}")
    print(f"Similarity           : {result['similarity']:.4f}")
    print(f"Edit script length   : {result['edit_script_length']}")
    print("=" * 60)

    # Print first few edit operations so you can inspect them
    preview_count = min(10, len(result["edit_script"]))
    print(f"\nShowing first {preview_count} edit operations:\n")

    for i, op in enumerate(result["edit_script"][:preview_count], start=1):
        print(f"{i}. {json.dumps(op, indent=2, ensure_ascii=False)}\n")

    # Save full diff to file
    out_name = f"diff_{country_a.replace(' ', '_')}_to_{country_b.replace(' ', '_')}.json"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    save_diff_json(result, out_path)

    print(f"Full diff saved to: {out_path}")


if __name__ == "__main__":
    main()