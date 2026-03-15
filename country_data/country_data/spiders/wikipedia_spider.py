import scrapy
import re


class CountrySpider(scrapy.Spider):
    name = "countries"
    start_urls = ["https://simple.wikipedia.org/wiki/List_of_countries"]

    SECTION_KEYWORDS = [
        "Area", "Population", "GDP", "Formation",
        "Legislature", "Government",
    ]

    def parse(self, response):
        """Extract all country links from the list page."""
        country_links = response.xpath(
            '//section[@id="mwBQ"]//a[@rel="mw:WikiLink"]/@href'
        ).getall()

        for link in country_links:
            if re.match(r"^//simple\.wikipedia\.org/wiki/[A-Z]", link):
                yield response.follow("https:" + link, callback=self.parse_country)

    def parse_country(self, response):
        """Extract infobox data from a country page."""
        country_name = response.css("span.mw-page-title-main::text").get("").strip()

        infobox = response.xpath("//table[contains(@class, 'infobox')]")
        if not infobox:
            self.logger.warning(f"No infobox found for {country_name}")
            return

        data = {
            "name": country_name,
            "url": response.url,
        }

        rows = infobox.css("tr")
        current_section = None

        for row in rows:
            header = row.css("th ::text").getall()
            value = row.css("td ::text").getall()

            if header and not value:
                section_name = self.clean_text(" ".join(header).strip())
                if not section_name:
                    continue

                is_section = any(
                    kw.lower() in section_name.lower()
                    for kw in self.SECTION_KEYWORDS
                )

                if is_section:
                    current_section = section_name
                    data[current_section] = {}
                else:
                    current_section = None
                    
            elif header and value:
                raw_key = " ".join(header).strip()
                key = self.clean_text(raw_key)
                val = self.clean_text(" ".join(value).strip())
                if not key or not val:
                    continue
                # Check if sub-row by looking for bullet in raw text
                is_sub_row = raw_key.strip().startswith("•")
                # Remove leading bullet from key
                key = re.sub(r"^•\s*", "", key)
                # GDP rows have both th and td but should start a new section
                is_section_with_value = any(
                    kw.lower() in key.lower() for kw in ["GDP"]
                )
                if is_section_with_value:
                    current_section = key
                    data[current_section] = {"estimate": val}
                elif (
                    is_sub_row
                    and current_section
                    and current_section in data
                    and isinstance(data[current_section], dict)
                ):
                    data[current_section][key] = val
                else:
                    current_section = None
                    data[key] = val
        # Remove empty sections
        data = {k: v for k, v in data.items() if v != {}}

        yield data

    def clean_text(self, text):
        # Remove CSS rules that leak in as text
        text = re.sub(r"\.mw-parser-output[^}]+\}", "", text)
        # Remove footnote references like [3], [a], [ 3 ], [ a ]
        text = re.sub(r"\[\s*\w+\s*\]", "", text)
        # Remove coordinate text
        text = re.sub(r"\d+°\d+[′']\s*[NSEW].*", "", text)
        # Remove leftover CSS class names
        text = re.sub(r"\.mw-[\w.-]+", "", text)
        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def close(self, reason):
        self.logger.info(f"Spider closed: {reason}")