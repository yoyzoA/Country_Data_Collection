"""Quick demo of the semantic/subsection-aware update costs added for Project 1."""

from country_data.semantic_similarity import semantic_update_cost


def token(label, context, parent_value=""):
    return {
        "label": label,
        "node_type": "token",
        "parent_label": context.split(" / ")[-1],
        "parent_context": context,
        "semantic_context": context,
        "parent_value": parent_value,
    }


def attribute(label, context=None):
    context = context or label
    ancestors = context.split(" / ")[:-1]
    return {
        "label": label,
        "node_type": "attribute",
        "ancestor_labels": ancestors,
        "parent_context": " / ".join(ancestors),
        "semantic_context": context,
    }


def main():
    examples = [
        (
            "4M vs 6M population",
            token("4", "Population / Total", "4 million"),
            token("6", "Population / Total", "6 million"),
        ),
        (
            "4M vs 100M population",
            token("4", "Population / Total", "4 million"),
            token("100", "Population / Total", "100 million"),
        ),
        (
            "1943 vs 1946 independence",
            token("1943", "Independence declared"),
            token("1946", "De facto Independence"),
        ),
        (
            "1943 vs 1776 independence",
            token("1943", "Independence declared"),
            token("1776", "Declaration"),
        ),
        (
            "95% vs 93% ethnicity",
            token("95%", "Ethnic groups"),
            token("93%", "Ethnic groups"),
        ),
        (
            "95% vs 30% ethnicity",
            token("95%", "Ethnic groups"),
            token("30%", "Ethnic groups"),
        ),
        (
            "Area Total vs GDP Total field",
            attribute("Total", "Area / Total"),
            attribute("Total", "GDP ( PPP ) / Total"),
        ),
        (
            "GDP PPP Total vs GDP nominal Total",
            attribute("Total", "GDP ( PPP ) / Total"),
            attribute("Total", "GDP (nominal) / Total"),
        ),
        (
            "Ethnic groups vs Federal Chancellor",
            attribute("Ethnic groups"),
            attribute("Federal Chancellor"),
        ),
    ]

    print("Semantic/subsection-aware update-cost demo")
    print("Lower cost = more similar. A cost of 2.0 means: use delete + insert instead of a misleading update.\n")
    for title, a, b in examples:
        print(f"{title:<42} cost = {semantic_update_cost(a, b):.4f}")


if __name__ == "__main__":
    main()
