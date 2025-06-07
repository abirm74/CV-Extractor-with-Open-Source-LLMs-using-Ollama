# cv_extractor_project/parsing/text_cleaner.py

import re
from datetime import datetime
from dateutil import parser as du_parser

# 1) ENHANCED BULLET‐LIKE CHARACTERS

# We expand the bullet equivalents to include: *, «, +, ° (degree sign), etc.
BULLET_EQUIVALENTS = [
    "\u2022",  # •
    "\u2023",  # ‣
    "\u2043",  # ⁃
    "\u2219",  # ∙
    "\u25E6",  # ◦
    "\u00BB",  # »
    "\u2013",  # – (en‐dash)
    "\u2014",  # — (em‐dash)
    "\uf0b7",  # ﬧ often in PDF bullets
    "\u25CF",  # ●
    "\u25CB",  # ○
    "*",       # ASCII asterisk used as bullet
    "+",       # plus sometimes used as bullet
    "°",       # degree sign often appears as OCR “•”
    "¬",       # sometimes OCR picks up “¬” instead of bullet
    "·",       # middle dot
    "»",       # angle quote
    "«",       # angle quote left
]


def normalize_bullets(text: str) -> str:
    """
    1) Replace any recognized bullet‐like character with a simple hyphen + space ("- ").
    2) Then remove lines that consist only of hyphens or hyphens + spaces (e.g. "----", "--  ", "-")
       so they don’t become empty bullet lines in the output.
    3) Collapse repeated hyphens into a single "- ".
    """
    # 1) Replace each bullet‐equivalent with "- "
    for bullet in BULLET_EQUIVALENTS:
        text = text.replace(bullet, "- ")

    # 2) Collapse any sequence of two or more hyphens into one "- "
    text = re.sub(r"-{2,}", "- ", text)

    # 3) Now remove lines that are only hyphens/spaces
    cleaned_lines = []
    for line in text.splitlines():
        # If, after stripping spaces and hyphens, nothing remains, skip the line
        if re.fullmatch(r"[-\s]+", line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # 4) Collapse multiple spaces or tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text


# 2) QUOTES & DASHES NORMALIZATION (unchanged)
def normalize_quotes_and_dashes(text: str) -> str:
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201C", '"').replace("\u201D", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)
    return text


# 3) PHONE NORMALIZATION WITH EXTRA SEPARATORS HANDLED

# Allow for "*" or "°" as a possible OCR‐inserted “separator” in phone
PHONE_REGEX = re.compile(
    r"""(?x)                    # verbose
    (?P<plus>\+)?               # optional leading plus
    [\s\(\)\-\.°\*]*            # separators: spaces, parentheses, hyphens, dots, degree, asterisk
    (?P<country>\d{1,3})?       # optional 1–3 digit country code
    [\s\(\)\-\.°\*]*            # more separators
    (?P<number>(?:\d[\s\-\.\(\)°\*]*){7,15})  # 7–15 digits with optional separators
    """
)


def normalize_phone_numbers(text: str) -> str:
    """
    Replace any matched phone‐like sequences with a cleaned version: +<country><digits> or <digits>.
    This handles OCR‐inserted •, °, or * inside the number.
    """
    def _replace(match):
        plus = "+" if match.group("plus") else ""
        country = match.group("country") or ""
        number = match.group("number") or ""
        # Strip all non‐digits from “number”
        digits = re.sub(r"\D", "", number)
        return f"{plus}{country}{digits}"

    return PHONE_REGEX.sub(_replace, text)


# 4) DATE NORMALIZATION (unchanged)
# We keep the same logic to normalize “Mai 2023 - June 2025” and “01/02/2018 - 02/02/2022”

MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "janvier": "01", "février": "02", "fevrier": "02", "mars": "03", "avril": "04", "mai": "05",
    "juin": "06", "juillet": "07", "août": "08", "aout": "08", "septembre": "09", "octobre": "10",
    "novembre": "11", "décembre": "12", "decembre": "12"
}

DATE_RANGE_REGEX = re.compile(
    r"""(?xi)
    (?P<start>
        (?:(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})       # e.g. 01/02/2018 or 1-2-18
        |
        (?:[A-Za-zéû]+(?:\s+\d{4}))                  # e.g. Mai 2023 or May 2023
        )
    )
    \s*[-–—]\s*
    (?P<end>
        (?:(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})
        |
        (?:[A-Za-zéû]+(?:\s+\d{4}))
        |
        (?:Present|present|présent|Présent)
        )
    )
    """
)

def _normalize_single_date(date_str: str) -> str:
    ds = date_str.strip()
    if not ds:
        return ""
    if ds.lower() in ("present", "présent"):
        return "Present"
    m = re.match(r"^(\d{1,2})[/\-](\d{2,4})$", ds)
    if m:
        mm, yy = m.group(1).zfill(2), m.group(2)
        if len(yy) == 2:
            yy = "20" + yy if int(yy) < 50 else "19" + yy
        return f"{mm}/{yy}"
    parts = ds.lower().split()
    if len(parts) == 2 and parts[0] in MONTH_MAP and re.fullmatch(r"\d{4}", parts[1]):
        return f"{MONTH_MAP[parts[0]]}/{parts[1]}"
    try:
        dt = du_parser.parse(ds, dayfirst=False, yearfirst=False)
        return dt.strftime("%m/%Y")
    except Exception:
        return date_str

def normalize_dates(text: str) -> str:
    def _replace_range(m):
        start_norm = _normalize_single_date(m.group("start"))
        end_norm = _normalize_single_date(m.group("end"))
        return f"{start_norm} - {end_norm}"

    text = DATE_RANGE_REGEX.sub(_replace_range, text)

    # Standalone single dates
    SINGLE_DATE_REGEX = re.compile(
        r"""(?xi)
        (?:
          (?:\d{1,2}[/\-]\d{2,4})
          |
          (?:[A-Za-zéû]+(?:\s+\d{4}))
        )
        """
    )
    text = SINGLE_DATE_REGEX.sub(lambda m: _normalize_single_date(m.group(0)), text)
    return text


# 5) FINAL CLEANUP

def final_cleanup(text: str) -> str:
    # Remove non‐printable/control characters (except newline and tab)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u0080-\u00FF]", "", text)
    # Collapse multiple blank lines into a single blank line
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # Trim trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text


# 6) COMPOSED CLEANER

# ────────────────────────────────────────────────────────────────────────────────
# 7) REMOVE RANDOM LEADING CHARS (©, ', ", etc.) & NORMALIZE TO BULLETS
# ────────────────────────────────────────────────────────────────────────────────

# Sometimes OCR spits out “©” or a stray quote at the start of a bullet line.
# We’ll treat any line that begins with one of these (© ' " `) as if it were a bullet.

LEADING_BULLET_EQUIVALENTS = [
    "©", "“", "”", "\"", "'", "`",   # stray quotes or copyright
    "·", "*", "+", "«", "»", "°", "–", "—"
]

def normalize_leading_bullets(text: str) -> str:
    """
    If a line starts with any of the LEADING_BULLET_EQUIVALENTS, replace that char
    with a single "- " so it becomes a uniform bullet. Then trim any leftover spaces.
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            # preserve blank lines
            cleaned.append("")
            continue

        # If the first character (ignoring whitespace) is in our list, replace it
        first = stripped[0]
        if first in LEADING_BULLET_EQUIVALENTS:
            # Drop leading bullet‐equivalent and any subsequent spaces
            rest = stripped[1:].lstrip()
            cleaned.append(f"- {rest}")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


# ────────────────────────────────────────────────────────────────────────────────
# 8) SEPARATE CONCATENATED FIELDS (PHONE, EMAIL, LINKEDIN, LOCATION)
# ────────────────────────────────────────────────────────────────────────────────

# If OCR or PyMuPDF concatenates phone/email/linkedin all on one line, we split them.
EMAIL_REGEX      = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
LINKEDIN_REGEX   = re.compile(r"(https?://(?:www\.)?linkedin\.com/[A-Za-z0-9/_\-]+)")
PHONE_REGEX_SPLIT= re.compile(r"(\+?\d{1,3}[\(\)\-\.\s°*]*\d{1,3}[\-\.\s°*]*\d{2,3}[\-\.\s°*]*\d{2,4})")
LOCATION_REGEX   = re.compile(r"([A-Za-z\s]+,\s*[A-Za-z]{2,})\s*$")  # e.g. Houston, Texas

def split_concatenated_fields(text: str) -> str:
    """
    For any line that contains multiple fields glued together (e.g. “Job Title12345551234addison@gmail.com=linkedin.com - Houston, Texas”),
    insert a newline before each detected email, phone, linkedin URL, and also separate trailing location if present.
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        # 1) Insert newline before email (if it’s not already on its own line)
        def _insert_newline_before_email(m):
            return "\n" + m.group(1)
        line = EMAIL_REGEX.sub(_insert_newline_before_email, line)

        # 2) Insert newline before linkedin URL
        def _insert_newline_before_linkedin(m):
            return "\n" + m.group(1)
        line = LINKEDIN_REGEX.sub(_insert_newline_before_linkedin, line)

        # 3) Insert newline before phone pattern
        def _insert_newline_before_phone(m):
            return "\n" + m.group(1)
        line = PHONE_REGEX_SPLIT.sub(_insert_newline_before_phone, line)

        # 4) If a location appears at the end (e.g. “- Houston, Texas”), ensure it’s on its own line
        m_loc = LOCATION_REGEX.search(line)
        if m_loc:
            loc = m_loc.group(1)
            # Remove the location from the end and append “\n<location>”
            line = LOCATION_REGEX.sub("", line).rstrip()
            line = f"{line}\n{loc}"

        # After inserting all newlines, split into sublines and re-add
        for sub in line.splitlines():
            cleaned.append(sub.rstrip())

    return "\n".join(cleaned)


# ────────────────────────────────────────────────────────────────────────────────
# 9) SPLIT SENTENCES WHEN BLOCKS ARE RUNNING TOGETHER (PyMuPDF fixes)
# ────────────────────────────────────────────────────────────────────────────────

# If PyMuPDF produces long paragraphs without hyphens/bullets, we can heuristically
# split on “. ” followed by a capital letter, or “. \n” cases.

def split_sentences(text: str) -> str:
    """
    Insert newline after each period that ends a sentence, if the next character is uppercase.
    Example:
      “... clinical trials.Collaborated ...” → “... clinical trials.\nCollaborated ...”
    """
    # 1) Insert newline between “.<space><Capital>”
    text = re.sub(r"\.\s+(?=[A-Z])", ".\n", text)

    # 2) Insert newline between “.<Capital>” (no space, missing space)
    text = re.sub(r"\.(?=[A-Z])", ".\n", text)

    return text


# ────────────────────────────────────────────────────────────────────────────────
# 10) UPDATED COMPOSED CLEANER
# ────────────────────────────────────────────────────────────────────────────────

def clean_extracted_text(raw_text: str) -> str:
    """
    1) Normalize any stray leading bullet-like characters  (©, *, etc.) → "- ".
    2) Normalize bullets & remove dash-only lines.
    3) Normalize quotes & dashes.
    4) Normalize phone numbers (handles "°" or "*" inside numbers).
    5) Normalize dates.
    6) Split concatenated fields (phone/email/linkedin/location) onto separate lines.
    7) Heuristically split sentences (for PyMuPDF blocks without bullets).
    8) Final cleanup (strip stray control chars, collapse blank lines, trim whitespace).
    """
    t = normalize_leading_bullets(raw_text)
    t = normalize_bullets(t)
    t = normalize_quotes_and_dashes(t)
    t = normalize_phone_numbers(t)
    t = normalize_dates(t)
    t = split_concatenated_fields(t)
    t = split_sentences(t)
    t = final_cleanup(t)
    return t