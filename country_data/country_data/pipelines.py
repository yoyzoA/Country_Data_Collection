"""
Scrapy pipeline for building document trees from scraped country data.

This pipeline collects all scraped items during the crawl, then when the
spider closes, it builds rooted ordered labeled trees for each country
and saves them to a JSON file.

To enable, add to settings.py:
    ITEM_PIPELINES = {
        'country_data.pipelines.TreeBuilderPipeline': 300,
    }
"""

import json
import os
from tree_builder import build_tree, build_all_trees


class TreeBuilderPipeline:

    def __init__(self):
        self.items = []

    def process_item(self, item, spider):
        """Collect each scraped country item."""
        self.items.append(dict(item))
        return item

    def close_spider(self, spider):
        """When spider finishes, build trees and save."""
        spider.logger.info(f"Building trees for {len(self.items)} countries...")

        # Build trees for all countries
        trees = build_all_trees(self.items)

        # Save tree representations as JSON
        output = []
        for name, tree in trees.items():
            output.append({
                "name": name,
                "node_count": tree.node_count(),
                "depth": tree.depth(),
                "tree": tree.to_dict(),
            })

        output_path = "country_trees.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        spider.logger.info(f"Saved {len(output)} trees to {output_path}")

        # Print summary
        spider.logger.info("=== Tree Summary ===")
        total_nodes = sum(t["node_count"] for t in output)
        avg_nodes = total_nodes / len(output) if output else 0
        avg_depth = sum(t["depth"] for t in output) / len(output) if output else 0
        spider.logger.info(f"Total countries: {len(output)}")
        spider.logger.info(f"Total nodes across all trees: {total_nodes}")
        spider.logger.info(f"Average nodes per tree: {avg_nodes:.1f}")
        spider.logger.info(f"Average tree depth: {avg_depth:.1f}")