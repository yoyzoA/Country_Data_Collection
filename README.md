# Country Data Collection and Tree Differencing

This project collects country infobox data from **Simple Wikipedia**, converts each country document into a **rooted ordered labeled tree**, compares pairs of trees using a **Chawathe-inspired Tree Edit Distance (TED)** workflow, applies the resulting **edit script** through **tree patching**, and reconstructs the patched tree back into **JSON** and **infobox-style text**.

## Project scope

The pipeline follows the five main stages of the course project:

1. **Data collection** from Wikipedia country infoboxes
2. **Pre-processing** into rooted ordered labeled trees
3. **Tree differencing** using a TED-inspired dynamic-programming comparison
4. **Document tree patching** using the extracted edit script
5. **Post-processing** back into JSON and readable infobox-style output

## Current implementation summary

- **Source website:** Simple Wikipedia country pages
- **Scraper:** `Scrapy`
- **Tree representation:** rooted ordered labeled tree
  - `root` node = country name
  - `attribute` nodes = flat key-value fields
  - `element` nodes = nested dictionary sections
  - `token` nodes = whitespace-tokenized value leaves
- **TED approach:** a **Chawathe-inspired** preorder LD-pair / sequence-alignment method
- **Patch operations:** `update`, `delete`, `insert`
- **Outputs:** diff JSON, patched tree JSON, reconstructed JSON, infobox-style text

## Important note about the TED method

This repository currently uses a **Chawathe-inspired approximation**, not a textbook ordered-tree TED implementation.

The comparison works by:

- flattening each tree in **preorder** into LD-pair-style entries
- running a **dynamic programming** alignment with insert / delete / update costs
- backtracking to recover an **edit script**

## Repository structure

```text
Country_Data_Collection/
├── README.md
└── country_data/
    ├── scrapy.cfg
    ├── countries.json
    ├── country_trees.json
    ├── run_ted.py
    ├── run_patch.py
    ├── run_postprocess.py
    ├── run_batch_test.py
    └── country_data/
        ├── pipelines.py
        ├── tree_builder.py
        ├── ted_diff.py
        ├── patch.py
        ├── post_process.py
        ├── settings.py
        └── spiders/
            └── wikipedia_spider.py
```

## What each file does

### Main scripts

- `country_data/run_ted.py`  
  Runs TED comparison for a selected country pair and saves the diff JSON.

- `country_data/run_patch.py`  
  Computes the edit script, applies patching to the source tree, verifies the patched tree against the target tree, and saves both the diff and the patched tree.

- `country_data/run_postprocess.py`  
  Reconstructs a tree back into JSON and infobox-style text. It can run on either an original country tree or a previously patched tree.

- `country_data/run_batch_test.py`  
  Runs the full pipeline on several predefined country pairs and reports summary metrics.

### Core modules

- `country_data/country_data/spiders/wikipedia_spider.py`  
  Scrapes the list of countries and extracts infobox data from each country page.

- `country_data/country_data/pipelines.py`  
  Collects scraped items and builds tree representations when the spider closes.

- `country_data/country_data/tree_builder.py`  
  Converts scraped JSON documents into rooted ordered labeled trees.

- `country_data/country_data/ted_diff.py`  
  Flattens trees in preorder, computes TED-style DP alignment, and extracts the edit script.

- `country_data/country_data/patch.py`  
  Applies the edit script to the source tree in a safe order.

- `country_data/country_data/post_process.py`  
  Converts a tree back into JSON and infobox-style text.

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
pip install scrapy
```

## How to run the project

All commands below should be run from:

```bash
cd country_data
```

### Step 1 — Scrape country data

```bash
scrapy crawl countries -o countries.json
```

This creates:

- `countries.json` — scraped country infobox data

### Step 2 — Build country trees automatically during scraping

If `TreeBuilderPipeline` is enabled in `country_data/settings.py`, the scraper will also generate:

- `country_trees.json` — rooted ordered labeled tree representation for each country

If needed, make sure this is enabled in `country_data/country_data/settings.py`:

```python
ITEM_PIPELINES = {
    'country_data.pipelines.TreeBuilderPipeline': 300,
}
```

### Step 3 — Compare two countries with TED

Edit the country names directly inside `run_ted.py`, or use the saved examples already produced in the folder.

Then run:

```bash
python run_ted.py
```

This prints:

- source / target node counts
- tree edit distance
- similarity score
- edit script length
- preview of the first edit operations

It also saves a diff file such as:

- `diff_France_to_Germany.json`

### Step 4 — Patch one country tree toward another

```bash
python run_patch.py Lebanon Switzerland
```

This:

- loads both trees from `country_trees.json`
- computes the edit script
- applies the patch to the source tree
- computes the **residual TED** between the patched tree and the real target tree
- saves:
  - `diff_Lebanon_to_Switzerland.json`
  - `patched_Lebanon_to_Switzerland.json`

### Step 5 — Post-process a tree back into JSON and text

#### Original tree

```bash
python run_postprocess.py Lebanon
```

This saves:

- `postprocessed_Lebanon.json`
- `postprocessed_Lebanon.txt`

#### Previously patched tree

```bash
python run_postprocess.py --patched Lebanon Switzerland
```

This saves:

- `postprocessed_patched_Lebanon_to_Switzerland.json`
- `postprocessed_patched_Lebanon_to_Switzerland.txt`

### Step 6 — Run the batch evaluation

```bash
python run_batch_test.py
```

This runs the full pipeline on predefined country pairs and reports:

- TED and similarity
- edit-operation counts
- cross-type update count
- residual TED after patching
- reconstructed field counts

## Tree-building rules

The current tree builder uses these rules:

- the **root** is the country name
- flat key-value pairs become **attribute** nodes
- nested dictionaries become **element** nodes
- text values are split into whitespace **token** leaves
- attribute nodes are sorted alphabetically
- element nodes preserve document order
- at each level, attributes are placed before elements

## Similarity formula

The TED similarity score is computed as:

```text
Sim(A, B) = 1 - TED(A*, B*) / (|A*| + |B*|)
```

where `A*` and `B*` are the preorder LD-pair-style sequences of the two trees.

## Patching strategy

The patcher applies operations in a safe order:

1. **UPDATEs first**
2. **DELETEs in reverse preorder** (deepest first)
3. **INSERTs in forward preorder** (shallowest first)

This helps avoid path invalidation while modifying tree structure.



