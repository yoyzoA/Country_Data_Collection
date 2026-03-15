# Country Data Scraper

Scrapes country information from [Simple Wikipedia](https://simple.wikipedia.org/wiki/List_of_countries) using Scrapy.

## Setup

```bash
pip install scrapy
scrapy startproject country_data
```

Copy `countries.py` into `country_data/country_data/spiders/`.

Add this to `country_data/country_data/settings.py`:

```python
DOWNLOAD_DELAY = 1
```

## Run

```bash
cd country_data
scrapy crawl countries -o countries.json
```

Output will be saved to `countries.json`.
