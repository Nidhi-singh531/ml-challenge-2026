"""Text normalisation and tokenisation for business entity resolution.

Deliberately language-agnostic: nothing here hard-codes US/India, because the
test set contains a third country (France) that never appears in training.
"""
import re
import unicodedata
from itertools import permutations

# Legal / filler tokens that carry almost no identity signal. They are kept in
# the token stream for blocking (they still help when a record is otherwise
# empty) but are down-weighted by IDF and stripped for the "core name" features.
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "l l c", "llp", "ltd", "limited", "limitee",
    "limitada", "private", "pvt", "pte", "corp", "corporation", "co", "company",
    "companies", "group", "holdings", "holding", "enterprises", "enterprise",
    "services", "service", "solutions", "society", "centre", "center", "trust",
    "associates", "assoc", "partners", "sarl", "sas", "sa", "eurl", "snc",
    "the", "and", "of", "for",
}

# Alias / "doing business as" connectors seen in the data: a record's name can be
# "<fake brand> DBA: <real name>" or "<real name> formerly known as <x>".
ALIAS_MARKERS = re.compile(
    r"\b(dba|d/b/a|aka|a/k/a|formerly known as|formerly|now known as|nee)\b[:\s]*",
    flags=re.I,
)

# Literal placeholders that appear as address text in the data. Left in, they
# create spurious "shared tokens" between two records that both lack an address.
NULL_TOKENS = {"null", "nan", "na", "none", "nil", "unknown", "n"}

# ``\w`` excludes combining marks, which in Indic scripts are the *vowels* — a
# plain ``[^\w]+`` split shatters every Devanagari/Bengali/Tamil word into
# consonant fragments. Keep the Indic block and the joiners inside tokens.
_TOKEN_SPLIT = re.compile(r"[^\w\u0900-\u0DFF\u200c\u200d]+", re.UNICODE)
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "8": "b"})


def strip_accents(s: str) -> str:
    """Drop combining marks from Latin letters only.

    Indic scripts encode their vowels as combining marks, so a blanket
    ``combining`` filter silently mangles every Devanagari/Bengali/Tamil name
    ('\u0930\u093f\u092f\u0932' -> '\u0930\u092f\u0932'). Restricting the strip to ASCII bases keeps
    'Br\u00e1nds' -> 'brands' while leaving those names intact.
    """
    out, base_is_latin = [], False
    for c in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(c):
            if not base_is_latin:
                out.append(c)
            continue
        base_is_latin = ("a" <= c.lower() <= "z")
        out.append(c)
    return "".join(out)


def normalize(s: str) -> str:
    """Lowercase, de-accent, collapse punctuation to spaces. Script preserved."""
    if not s:
        return ""
    # NFKC and accent stripping are no-ops on ASCII, which is most of the data;
    # skipping them makes tokenising the 10M-record target pool ~3x faster.
    if s.isascii():
        s = s.lower()
    else:
        s = strip_accents(unicodedata.normalize("NFKC", s)).lower()
    s = _TOKEN_SPLIT.sub(" ", s)
    return " ".join(s.split())


def tokens(s: str):
    return [t for t in normalize(s).split() if t not in NULL_TOKENS]


def unleet(tok: str) -> str:
    """'8rands' -> 'brands', 'visi0n' -> 'vision', 'c0m' -> 'com'.

    Only applied to tokens that mix letters and digits, so house numbers
    ('17337') are left alone.
    """
    if tok.isdigit() or tok.isalpha():
        return tok
    if any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok):
        return tok.translate(_LEET)
    return tok


def name_tokens(name: str):
    """Tokens of a business name, alias markers removed, leet-speak repaired."""
    name = ALIAS_MARKERS.sub(" ", name or "")
    name = re.sub(r"\.(com|net|org|in|co|io|fr)\b", " ", name, flags=re.I)
    name = name.replace("@", " ")
    return [unleet(t) for t in tokens(name)]


def core_name_tokens(name: str):
    """Name tokens with legal suffixes dropped — the identity-bearing part."""
    return [t for t in name_tokens(name) if t not in LEGAL_SUFFIXES]


def addr_tokens(addr: str):
    return tokens(addr)


def numeric_tokens(toks):
    """House / plot numbers, normalised for the leading-digit noise in the data.

    '17337' vs '7337' vs '0669' vs '669' all refer to the same building in this
    dataset, so we compare on the stripped tail as well as the raw form.
    """
    out = set()
    for t in toks:
        if t.isdigit():
            out.add(t.lstrip("0") or "0")
            if len(t) > 3:
                out.add(t[1:].lstrip("0") or "0")  # dropped/added leading digit
    return out


def script_of(s: str) -> str:
    """Coarse script label: 'latin', 'indic', 'other', 'none'."""
    latin = indic = 0
    for c in s:
        o = ord(c)
        if ("a" <= c.lower() <= "z"):
            latin += 1
        elif 0x0900 <= o <= 0x0DFF:
            indic += 1
    if latin == 0 and indic == 0:
        return "none"
    return "latin" if latin >= indic else "indic"


# --------------------------------------------------------------------------- #
# mechanical romanisation + phonetic skeletons
# --------------------------------------------------------------------------- #
# The alias lexicon only covers words frequent enough to be mined; proper names
# ('शिव' / 'Shiva', 'প্রোডাক্টস' / 'Products') need a generic bridge. Every
# Brahmic script in U+0900-U+0DFF is described by the stdlib Unicode database in
# the same vocabulary ('DEVANAGARI LETTER SHA', 'TAMIL VOWEL SIGN I', '... SIGN
# VIRAMA'), so one rule set romanises all of them without any per-language table.
_INDIC_SIGN = {"VIRAMA": "", "NUKTA": "", "ANUSVARA": "n", "CANDRABINDU": "n",
               "VISARGA": "h", "AVAGRAHA": "", "AI LENGTH MARK": "",
               "AU LENGTH MARK": ""}


def _romanize_char(c):
    """-> (kind, latin) with kind in {'cons', 'vsign', 'virama', 'other'}."""
    try:
        nm = unicodedata.name(c)
    except ValueError:
        return "other", ""
    if " LETTER " in nm:
        sound = nm.split(" LETTER ", 1)[1].split()[-1].lower()
        if sound in ("a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo",
                     "au") or " VOCALIC " in nm:
            return "other", sound[:1] if " VOCALIC " not in nm else "r"
        return "cons", sound
    if " VOWEL SIGN " in nm:
        s = nm.split(" VOWEL SIGN ", 1)[1].lower()
        return "vsign", "r" if "vocalic" in s else s.split()[-1]
    if " SIGN " in nm:
        s = nm.split(" SIGN ", 1)[1]
        if s == "VIRAMA":
            return "virama", ""
        return "other", _INDIC_SIGN.get(s, "")
    if " DIGIT " in nm:
        return "other", str(unicodedata.digit(c, ""))
    return "other", ""


_ROMAN_CACHE: dict = {}


def romanize(tok: str) -> str:
    """Rough Latin rendering of an Indic token; other tokens pass through.

    Consonants carry the inherent 'a' unless a vowel sign or virama follows —
    exactly the abugida rule — so 'शिव' -> 'shiva', 'प्रोडक्ट्स' -> 'proddakttsa'.
    Precision is irrelevant here: the output only feeds ``skeleton``.
    """
    if tok.isascii():
        return tok
    r = _ROMAN_CACHE.get(tok)
    if r is not None:
        return r
    out = []
    pending_a = False
    for c in tok:
        if not (0x0900 <= ord(c) <= 0x0DFF):
            if pending_a:
                out.append("a")
                pending_a = False
            if c not in "‌‍":
                out.append(c)
            continue
        kind, lat = _romanize_char(c)
        if kind == "cons":
            if pending_a:
                out.append("a")
            out.append(lat.rstrip("a") or lat)
            pending_a = True
        elif kind == "vsign":
            out.append(lat)
            pending_a = False
        elif kind == "virama":
            pending_a = False
        else:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(lat)
    if pending_a:
        out.append("a")
    r = "".join(out)
    if len(_ROMAN_CACHE) < 2_000_000:
        _ROMAN_CACHE[tok] = r
    return r


# Consonant classes that transliteration and typing noise blur together.
_SKEL = str.maketrans({"c": "k", "q": "k", "g": "k", "j": "k", "d": "t", "b": "p",
                       "f": "p", "v": "p", "w": "p", "z": "s", "x": "s", "y": "",
                       "h": ""})
_VOWELS = re.compile(r"[aeiou]+")


def skeleton(tok: str) -> str:
    """Script-independent consonant skeleton: 'Products', 'प्रोडक्ट्स' -> 'prtkts'."""
    s = _VOWELS.sub("", romanize(tok).translate(_SKEL))
    out: list = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def name_keys(name: str, amap=None):
    """Blocking keys of the *name channel* (see blocking.py).

    * the canonical name tokens themselves;
    * ``#`` + concatenations, so a web-ified name ('securecloudservices.com',
      'APPLIANCEPLATINUM.COM', 'vrikshammemorialtrustcom') meets its spaced
      form: every prefix run of the tokens, and every order of <= 3 core tokens;
    * ``~`` + consonant skeletons, so 'ರಾಮ್ ಎನರ್ಜಿ' meets 'Ram Energy' even when
      the alias lexicon has never seen the words.
    """
    toks = canonical(name_tokens(name), amap)
    keys = set(toks)
    if not toks:
        return keys
    if len(toks) == 1:
        t = toks[0]
        keys.add("#" + t)
        if t.endswith("com") and len(t) > 6:
            keys.add("#" + t[:-3])
    else:
        run = toks[0]
        for t in toks[1:6]:
            run += t
            keys.add("#" + run)
        core = [t for t in toks if t not in LEGAL_SUFFIXES]
        if 2 <= len(core) <= 3:
            keys.update("#" + "".join(p) for p in permutations(core))
    for t in toks:
        s = skeleton(t)
        if len(s) >= 2:
            keys.add("~" + s)
    return keys


def char_ngrams(s: str, n: int = 3):
    s = normalize(s).replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def containment(a: set, b: set) -> float:
    """|a & b| / min(|a|,|b|) — robust to one side being truncated."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# --------------------------------------------------------------------------- #
# alias lexicon (transliteration + abbreviation), mined from the training data
# --------------------------------------------------------------------------- #
def load_aliases(path):
    """Read a two-column TSV (variant<TAB>canonical) into a dict."""
    amap = {}
    if not path:
        return amap
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if first and "\t" in first and not first.lower().startswith("variant"):
            v, c = first.rstrip("\n").split("\t")[:2]
            amap[v] = c
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                amap[parts[0]] = parts[1]
    return amap


def canonical(toks, amap):
    """Map each token through the alias lexicon (identity when absent)."""
    if not amap:
        return list(toks)
    return [amap.get(t, t) for t in toks]
