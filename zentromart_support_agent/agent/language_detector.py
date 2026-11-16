"""Language detection module for ZentroMart AI Customer Support Agent.

This module detects customer communication languages without third-party ML models,
utilizing fast Unicode code-point range inspections and curated keyword heuristics.

Supported Languages:
    - en: English (default fallback for unrecognized or generic Latin text)
    - hi: Hindi (Devanagari script U+0900 - U+097F)
    - hi-en: Hinglish (Hindi written in Latin script with words like kya, hai, mera, etc.)
    - mr: Marathi (Devanagari script with Marathi keywords like aahe/mala/kasa, or Latin Marathi)
    - ta: Tamil (Tamil script U+0B80 - U+0BFF)
    - kn: Kannada (Kannada script U+0C80 - U+0CFF)
    - te: Telugu (Telugu script U+0C00 - U+0C7F)
    - ml: Malayalam (Malayalam script U+0D00 - U+0D7F)
    - bn: Bengali (Bengali script U+0980 - U+09FF)
    - gu: Gujarati (Gujarati script U+0A80 - U+0AFF)

Policy & Design References:
    - ZentroMart Knowledge Base: [STY-4] Language policy ("Reply in the language the customer uses").
    - Pure Python implementation: zero external dependencies, deterministic, low-latency.
"""

import re
from typing import Dict, Final, FrozenSet, List, Optional, Set, Tuple

# Supported ISO 639-1 / IETF language tags
SUPPORTED_LANGUAGES: Final[Tuple[str, ...]] = (
    "en",
    "hi",
    "hi-en",
    "ta",
    "kn",
    "te",
    "ml",
    "bn",
    "gu",
    "mr",
)

# Human-readable mapping of language codes to display names
LANGUAGE_NAMES: Final[Dict[str, str]] = {
    "en": "English",
    "hi": "Hindi",
    "hi-en": "Hinglish",
    "ta": "Tamil",
    "kn": "Kannada",
    "te": "Telugu",
    "ml": "Malayalam",
    "bn": "Bengali",
    "gu": "Gujarati",
    "mr": "Marathi",
}

# Unicode code-point ranges for Indic scripts
# Each tuple defines (inclusive_start, inclusive_end) in Unicode space
SCRIPT_RANGES: Final[Dict[str, Tuple[int, int]]] = {
    "bn": (0x0980, 0x09FF),  # Bengali
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "devanagari": (0x0900, 0x097F),  # Devanagari (Hindi and Marathi)
}

# Pre-compiled regular expressions for fast Indic character matching
RE_DEVANAGARI: Final[re.Pattern] = re.compile(r"[\u0900-\u097F]")
RE_BENGALI: Final[re.Pattern] = re.compile(r"[\u0980-\u09FF]")
RE_GUJARATI: Final[re.Pattern] = re.compile(r"[\u0A80-\u0AFF]")
RE_TAMIL: Final[re.Pattern] = re.compile(r"[\u0B80-\u0BFF]")
RE_TELUGU: Final[re.Pattern] = re.compile(r"[\u0C00-\u0C7F]")
RE_KANNADA: Final[re.Pattern] = re.compile(r"[\u0C80-\u0CFF]")
RE_MALAYALAM: Final[re.Pattern] = re.compile(r"[\u0D00-\u0D7F]")

# Tokenizer for words
RE_WORDS_LATIN: Final[re.Pattern] = re.compile(r"[a-zA-Z]+")
RE_WORDS_DEVANAGARI: Final[re.Pattern] = re.compile(r"[\u0900-\u097F]+")

# ============================================================================
# Marathi (mr) Keywords & Script Markers
# ============================================================================

# Marathi specific characters in Devanagari script:
# U+0933: ळ (Devanagari letter LLA - highly characteristic of Marathi, absent in modern standard Hindi)
# U+0931: ऱ (Devanagari letter RRA / eyelash reph)
MARATHI_DISTINCT_CHARS: Final[FrozenSet[str]] = frozenset({"ळ", "ऱ"})

# Devanagari Marathi distinctive vocabulary:
# Words like आहे (aahe), मला (mala), कसा (kasa), काय (kaay), कुठे (kuthe), कधी (kadhi), etc.
MARATHI_DEVANAGARI_KEYWORDS: Final[FrozenSet[str]] = frozenset(
    {
        # Common verbs & auxiliaries
        "आहे",
        "आहेत",
        "आहेस",
        "आहोत",
        "नाही",
        "नाहीत",
        "नाहि",
        "नसेल",
        "नसतात",
        "होय",
        "होता",
        "होती",
        "होते",
        "होणार",
        "होईल",
        "झाला",
        "झाली",
        "झाले",
        "झालीय",
        "झालाय",
        "करा",
        "करावे",
        "करतो",
        "करते",
        "करतात",
        "करायचा",
        "करायची",
        "करायचे",
        "करायचेय",
        "केला",
        "केली",
        "केले",
        "करू",
        "करूया",
        "द्या",
        "द्यावे",
        "घ्या",
        "पाहिजे",
        "हवे",
        "हवा",
        "हवी",
        "सांगा",
        "सांग",
        "मिळेल",
        "मिळाले",
        "मिळाला",
        "मिळाली",
        # Pronouns
        "मला",
        "तुला",
        "आम्हाला",
        "तुम्हाला",
        "माझा",
        "माझी",
        "माझे",
        "माझ्या",
        "तुमचा",
        "तुमची",
        "तुमचे",
        "तुमच्या",
        "आपला",
        "आपली",
        "आपले",
        "आपल्या",
        "त्याचा",
        "त्याची",
        "त्याचे",
        "त्यांना",
        "त्यांच्या",
        "हे",
        "ह्या",
        # Question words
        "कसा",
        "कशी",
        "कसे",
        "कशा",
        "कसं",
        "काय",
        "कुठे",
        "कधी",
        "का",
        "कशासाठी",
        "किती",
        # Particles & adverbs
        "परत",
        "किंवा",
        "पण",
        "नंतर",
        "आता",
        "खूप",
        "कमी",
        "लवकर",
        "काही",
        "अजून",
    }
)

# Marathi postpositions/suffixes in Devanagari often appended to nouns/pronouns
MARATHI_DEVANAGARI_SUFFIXES: Final[Tuple[str, ...]] = (
    "च्या",
    "ची",
    "चे",
    "चा",
    "मध्ये",
    "वरून",
    "कडून",
    "साठी",
    "पर्यंत",
    "बद्दल",
)

# Latin script Marathi keywords (e.g., aahe, mala, kasa)
MARATHI_LATIN_KEYWORDS: Final[FrozenSet[str]] = frozenset(
    {
        "aahe",
        "aahet",
        "ahe",
        "ahet",
        "mala",
        "tula",
        "aamhala",
        "tumhala",
        "kasa",
        "kashi",
        "kase",
        "kasam",
        "kuthe",
        "kadhi",
        "karaycha",
        "karaychi",
        "karayche",
        "karto",
        "karte",
        "kartat",
        "pahije",
        "parat",
        "jhala",
        "jhali",
        "jhale",
        "zala",
        "zali",
        "zale",
        "majha",
        "majhi",
        "majhe",
        "maza",
        "mazi",
        "maze",
        "tumcha",
        "tumchi",
        "tumche",
        "saanga",
        "milnar",
        "milala",
        "milali",
        "milale",
        "yenar",
        "havay",
        "havi",
    }
)

# ============================================================================
# Hinglish (hi-en) Keywords
# ============================================================================

# Curated Romanized Hindi words that do NOT collide with standard English words.
# Includes question words, pronouns, verbs, auxiliaries, customer service terms.
HINGLISH_KEYWORDS: Final[FrozenSet[str]] = frozenset(
    {
        # Question words
        "kya",
        "kyu",
        "kyun",
        "kyon",
        "kyunki",
        "kyoki",
        "kab",
        "kahan",
        "kaha",
        "kidhar",
        "kaise",
        "kaisa",
        "kaisi",
        "kitna",
        "kitne",
        "kitni",
        "kaun",
        "kaunsa",
        "kaunsi",
        "kaunse",
        # Pronouns & demonstratives
        "mera",
        "meri",
        "mere",
        "mujhe",
        "mujhko",
        "maine",
        "tera",
        "teri",
        "tere",
        "tujhe",
        "tujhko",
        "tumhara",
        "tumhari",
        "tumhare",
        "tumko",
        "tumhe",
        "aapka",
        "aapki",
        "aapke",
        "aapko",
        "hamara",
        "hamari",
        "hamare",
        "hume",
        "humein",
        "humko",
        "iska",
        "iski",
        "iske",
        "isko",
        "ise",
        "isi",
        "uska",
        "uski",
        "uske",
        "usko",
        "usse",
        "inka",
        "inki",
        "inke",
        "inko",
        "inhe",
        "inhein",
        "unka",
        "unki",
        "unke",
        "unko",
        "unhe",
        "unhein",
        "apna",
        "apni",
        "apne",
        "kisi",
        "kisko",
        "kisse",
        # Verbs & auxiliaries
        "hai",
        "hain",
        "hoon",
        "hun",
        "tha",
        "thi",
        "hoga",
        "hogi",
        "honge",
        "hona",
        "hone",
        "hota",
        "hoti",
        "hote",
        "hue",
        "hui",
        "hua",
        "aayega",
        "aayegi",
        "aayenge",
        "aaya",
        "aayi",
        "aaye",
        "aata",
        "aati",
        "aate",
        "aana",
        "aane",
        "jaayega",
        "jaayegi",
        "jaayenge",
        "jaana",
        "jaane",
        "jaata",
        "jaati",
        "jaate",
        "gaya",
        "gayi",
        "gaye",
        "karein",
        "karna",
        "karo",
        "karta",
        "karti",
        "karte",
        "karunga",
        "karungi",
        "karenge",
        "kar",
        "kiya",
        "kiye",
        "kiyi",
        "milega",
        "milegi",
        "milenge",
        "mila",
        "mili",
        "mile",
        "milta",
        "milti",
        "milte",
        "milna",
        "chahiye",
        "chahie",
        "sakta",
        "sakti",
        "sakte",
        "sake",
        "dena",
        "dedo",
        "dijiye",
        "diya",
        "diye",
        "dega",
        "degi",
        "lena",
        "lelo",
        "lijiye",
        "liya",
        "liye",
        "lega",
        "legi",
        "dekho",
        "dekhiye",
        "dekhna",
        "batao",
        "bataiye",
        "batana",
        "rakho",
        "rakhiye",
        "rakhna",
        "samajh",
        "samajhna",
        "raha",
        "rahi",
        "rahe",
        "laga",
        "lagi",
        "lage",
        "lagta",
        "lagti",
        "lagte",
        "shuru",
        "khatam",
        # Adverbs, particles, time & customer service terms
        "nahi",
        "nahin",
        "wapas",
        "vaapas",
        "paisa",
        "paise",
        "rupaye",
        "rupay",
        "rupya",
        "abhi",
        "tabhi",
        "kabhi",
        "baare",
        "parso",
        "wala",
        "wali",
        "wale",
        "sawaal",
        "jawaab",
        "theek",
        "thik",
        "teesri",
        "pehla",
        "pehli",
        "doosra",
        "doosri",
        "dusra",
        "dusri",
        "bohot",
        "bahut",
        "thoda",
        "thodi",
        "thode",
        "zyada",
        "jyada",
        "sirf",
        "lekin",
        "magar",
        "isliye",
        "dhanyawad",
        "shukriya",
        "namaste",
        "namaskar",
        "kripya",
        "kripaya",
        "madad",
        "insaan",
        "kharida",
        "mangwaya",
        "mangwayi",
        "hafte",
        "hafta",
        "mahina",
        "mahine",
        "agle",
        "agla",
        "agli",
        "aur",
        "mein",
        "pe",
    }
)


def get_indic_script_counts(text: str) -> Dict[str, int]:
    """Count characters corresponding to each recognized Indic script block.

    Args:
        text: Input string to inspect.

    Returns:
        Dictionary mapping script identifiers ('bn', 'gu', 'ta', 'te', 'kn', 'ml', 'devanagari')
        to the count of matched characters.
    """
    return {
        "devanagari": len(RE_DEVANAGARI.findall(text)),
        "bn": len(RE_BENGALI.findall(text)),
        "gu": len(RE_GUJARATI.findall(text)),
        "ta": len(RE_TAMIL.findall(text)),
        "te": len(RE_TELUGU.findall(text)),
        "kn": len(RE_KANNADA.findall(text)),
        "ml": len(RE_MALAYALAM.findall(text)),
    }


def is_marathi_devanagari(text: str) -> bool:
    """Determine whether Devanagari text contains distinctive Marathi characteristics.

    Checks for:
        1. Characteristic Marathi characters like ळ (LLA U+0933) or ऱ (RRA U+0931).
        2. Known Marathi Devanagari keywords (e.g., आहे, मला, कसा, कुठे, कधी, परत, पाहिजे).
        3. Distinctive Marathi grammatical postpositions (च्या, ची, चे, चा, मध्ये, साठी, etc.).
        4. Transliterated Marathi keywords (aahe, mala, kasa) if mixed with Devanagari.

    Args:
        text: Devanagari script text.

    Returns:
        True if text matches Marathi heuristics, False otherwise.
    """
    # 1. Distinctive Marathi characters
    for ch in MARATHI_DISTINCT_CHARS:
        if ch in text:
            return True

    # 2. Devanagari word tokens
    words = RE_WORDS_DEVANAGARI.findall(text)
    word_set = set(words)
    if word_set & MARATHI_DEVANAGARI_KEYWORDS:
        return True

    # 3. Postposition suffix heuristics on Devanagari tokens
    for w in words:
        for suffix in MARATHI_DEVANAGARI_SUFFIXES:
            if w.endswith(suffix) and len(w) > len(suffix) + 1:
                return True

    # 4. Transliterated Marathi keywords in mixed text
    latin_words = set(RE_WORDS_LATIN.findall(text.lower()))
    if latin_words & MARATHI_LATIN_KEYWORDS:
        return True

    return False


def is_marathi_latin(text: str) -> bool:
    """Determine whether Latin-script text contains distinctive Marathi keywords.

    Heuristics check for Marathi keywords (aahe, mala, kasa, etc.) as noted in prompt:
    "mr (Marathi - Devanagari with Marathi keywords like aahe/mala/kasa)".

    Args:
        text: Latin script text.

    Returns:
        True if text matches Latin Marathi heuristics, False otherwise.
    """
    tokens = set(RE_WORDS_LATIN.findall(text.lower()))
    if not tokens:
        return False

    # Immediate matches for strong Marathi distinctive words
    strong_marathi = {
        "aahe",
        "aahet",
        "kasa",
        "kashi",
        "kase",
        "kuthe",
        "kadhi",
        "karaycha",
        "karaychi",
        "karayche",
        "pahije",
        "parat",
        "jhala",
        "jhali",
        "jhale",
        "zala",
        "zali",
        "zale",
        "tumcha",
        "tumchi",
        "tumche",
        "majha",
        "majhi",
        "majhe",
    }
    if tokens & strong_marathi:
        return True

    # 'mala' accompanied by other Marathi indicators or romanized Marathi verbs
    if "mala" in tokens and (tokens & MARATHI_LATIN_KEYWORDS):
        return True

    return False


def is_hinglish(text: str) -> bool:
    """Determine whether Latin-script text contains Hindi words (Hinglish).

    Args:
        text: Latin script text.

    Returns:
        True if text contains one or more recognized Hinglish keywords, False otherwise.
    """
    tokens = set(RE_WORDS_LATIN.findall(text.lower()))
    return bool(tokens & HINGLISH_KEYWORDS)


def detect_language(text: str) -> str:
    """Detect the language of customer text.

    Supported Language Codes:
        - 'en': English (default fallback)
        - 'hi': Hindi (Devanagari script U+0900 - U+097F)
        - 'hi-en': Hinglish (Latin script Hindi words like kya/hai/mera/kab/aayega)
        - 'mr': Marathi (Devanagari script with Marathi keywords like aahe/mala/kasa, or Latin Marathi)
        - 'ta': Tamil (Tamil script U+0B80 - U+0BFF)
        - 'kn': Kannada (Kannada script U+0C80 - U+0CFF)
        - 'te': Telugu (Telugu script U+0C00 - U+0C7F)
        - 'ml': Malayalam (Malayalam script U+0D00 - U+0D7F)
        - 'bn': Bengali (Bengali script U+0980 - U+09FF)
        - 'gu': Gujarati (Gujarati script U+0A80 - U+0AFF)

    Algorithm:
        1. Input validation: Non-string, None, or whitespace-only strings default to 'en'.
        2. Unicode range checks: Count characters across Indic scripts.
        3. If Indic script characters are detected:
            - If Devanagari is dominant: check for Marathi markers (e.g. ळ, आहे, मला, kasa).
              If matched, return 'mr'; else return 'hi'.
            - If another Indic script is dominant (ta, kn, te, ml, bn, gu), return that code.
        4. If no Indic script characters are detected:
            - Check for Latin Marathi keywords ('aahe', 'mala', 'kasa', etc.) -> return 'mr'.
            - Check for Hinglish keywords ('kya', 'hai', 'mera', 'aayega', etc.) -> return 'hi-en'.
            - Default to 'en'.

    Args:
        text: The user query, message, or transcript text to analyze.

    Returns:
        A two-letter language code or 'hi-en'.
    """
    if not isinstance(text, str):
        return "en"

    cleaned_text = text.strip()
    if not cleaned_text:
        return "en"

    # Step 1: Check Indic script character counts
    script_counts = get_indic_script_counts(cleaned_text)
    dominant_script, max_count = max(script_counts.items(), key=lambda item: item[1])

    if max_count > 0:
        if dominant_script == "devanagari":
            # Disambiguate Hindi vs Marathi
            if is_marathi_devanagari(cleaned_text):
                return "mr"
            return "hi"
        return dominant_script

    # Step 2: No Indic script found. Check Latin script heuristics.
    # Check for Latin Marathi first (e.g. "aahe mala kasa", "mala refund kadhi milnar aahe")
    if is_marathi_latin(cleaned_text):
        return "mr"

    # Check for Hinglish keywords (e.g. "Order number hai 8978 2837", "Mera refund kahan hai?")
    if is_hinglish(cleaned_text):
        return "hi-en"

    # Default to English
    return "en"
