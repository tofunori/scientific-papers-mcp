"""Regex patterns for metadata extraction from markdown documents"""

import re
from typing import Dict, List, Pattern

# Thematic keywords for glacier albedo research
THEMATIC_TAGS = {
    "brdf": [
        "BRDF",
        "bidirectional",
        "reflectance",
        "directional",
        "anisotropic",
    ],
    "albedo": [
        "albedo",
        "albédo",
        "reflectance",
        "snow albedo",
        "ice albedo",
        "surface reflectance",
    ],
    "aerosols": [
        "aerosol",
        "aérosol",
        "black carbon",
        "carbon noir",
        "dust",
        "poussière",
        "LAP",
        "light-absorbing",
        "soot",
        "suie",
    ],
    "glaciers": [
        "glacier",
        "ice sheet",
        "snowpack",
        "snow",
        "neige",
        "calotte",
        "inlandsis",
    ],
    "remote_sensing": [
        "satellite",
        "remote sensing",
        "télédétection",
        "imaging",
        "spectral",
    ],
}

# Instruments and datasets
INSTRUMENTS = {
    "modis": r"\bMODIS\b",
    "sentinel": r"\bSentinel[-\s]?[12]\b",
    "landsat": r"\bLandsat[-\s]?[8-9]\b",
    "avhrr": r"\bAVHRR\b",
    "viirs": r"\bVIIRS\b",
    "meris": r"\bMERIS\b",
    "gedi": r"\bGEDI\b",
    "icesat": r"\bICESat[-\s]?2?\b",
}

# Regex patterns for metadata extraction
METADATA_PATTERNS: Dict[str, Pattern] = {
    # Title - usually first H1
    "title": re.compile(r"^#\s+(.+?)(?:\n|$)", re.MULTILINE),

    # Authors - various formats
    "authors": re.compile(
        r"(?:\*\*Authors?:\*\*\s*|Authors?:\s*)(.+?)(?:\n|\*\*|$)",
        re.IGNORECASE
    ),

    # Journal/Publication
    "journal": re.compile(
        r"(?:\*\*Journal:\*\*\s*|Journal:\s*|Published in:?\s*)(.+?)(?:\n|\*\*|$)",
        re.IGNORECASE
    ),

    # DOI
    "doi": re.compile(
        r"(?:\*\*DOI:\*\*\s*|DOI:\s*)(https?://doi\.org/[\w\./\-\(\)]+)",
        re.IGNORECASE
    ),

    # Publication year
    "year": re.compile(
        r"(?:Publication Date|Published|Year).*?(\d{4})",
        re.IGNORECASE
    ),

    # Abstract - usually in ## Abstract section
    "abstract": re.compile(
        r"##\s*Abstract\s*\n\n(.+?)(?=\n##|\Z)",
        re.IGNORECASE | re.DOTALL
    ),
}


def compile_patterns() -> Dict[str, Pattern]:
    """Return compiled regex patterns"""
    return METADATA_PATTERNS


def extract_year_from_filename(filename: str) -> int | None:
    """
    Try to extract year from filename
    Common patterns: "author-2021-title", "2021_paper", "paper_2021"
    """
    year_match = re.search(r"(\d{4})", filename)
    if year_match:
        year = int(year_match.group(1))
        if 1950 <= year <= 2050:  # Reasonable bounds
            return year
    return None


def extract_tags_from_text(text: str) -> List[str]:
    """
    Extract thematic tags from document text based on keywords
    """
    tags = []
    text_lower = text.lower()

    for tag, keywords in THEMATIC_TAGS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                tags.append(tag)
                break  # Only add tag once

    return list(set(tags))  # Remove duplicates


def extract_instruments_from_text(text: str) -> List[str]:
    """
    Extract instruments/datasets mentioned in text
    """
    instruments = []

    for instrument, pattern in INSTRUMENTS.items():
        if re.search(pattern, text):
            instruments.append(instrument)

    return instruments


def extract_authors_list(authors_string: str) -> List[str]:
    """
    Parse author string into list of authors
    Handles formats like: "Smith, J., Doe, A." or "Smith J, Doe A"
    """
    if not authors_string:
        return []

    # Split by common separators
    authors = re.split(r"[,;]\s*(?:and\s+)?", authors_string)

    # Clean up each author
    authors = [
        author.strip().replace(" et al.", "").strip()
        for author in authors
        if author.strip()
    ]

    return authors


def clean_text(text: str) -> str:
    """Clean text for processing"""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove markdown formatting
    text = re.sub(r"[*_`]", "", text)
    return text.strip()
