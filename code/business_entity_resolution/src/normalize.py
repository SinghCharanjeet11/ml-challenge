"""Text normalisation and tokenisation, expressed as polars operations."""
import polars as pl

# Street / address words -> canonical short form (applied per token).
ADDR_CANON = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "drive": "dr", "lane": "ln", "court": "ct",
    "parkway": "pkwy", "pky": "pkwy", "place": "pl", "circle": "cir", "cove": "cv", "highway": "hwy",
    "terrace": "ter", "square": "sq", "suite": "ste", "apartment": "apt", "building": "bldg",
    "floor": "fl", "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt", "fort": "ft",
    "trail": "trl", "way": "wy", "expressway": "expy", "freeway": "fwy", "center": "ctr", "centre": "ctr",
    "point": "pt", "heights": "hts", "junction": "jct", "crossing": "xing", "ridge": "rdg",
    "alley": "aly", "path": "path", "loop": "loop", "pike": "pike", "run": "run",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
    "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    # French
    "r": "rue", "allee": "all", "impasse": "imp", "chemin": "ch", "route": "rte", "cours": "crs",
    "quai": "qu", "residence": "res", "bis": "bis",
    # India
    "nagar": "ngr", "marg": "mg", "colony": "col", "sector": "sec", "near": "nr", "opposite": "opp",
    "opp": "opp", "bombay": "mumbai", "gurgaon": "gurugram", "bangalore": "bengaluru", "calcutta": "kolkata",
    "madras": "chennai", "keralam": "kerala",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "ladakh": "la",
}

NAME_CANON = {
    "private": "pvt", "pvt": "pvt", "limited": "ltd", "ltd": "ltd", "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc", "company": "co", "co": "co", "llc": "llc", "llp": "llp",
    "and": "&", "&": "&", "the": "the", "international": "intl", "intl": "intl", "services": "service",
    "technologies": "technology", "tech": "technology", "enterprises": "enterprise", "brothers": "bros",
    "shri": "shree", "sri": "shree", "shree": "shree", "associates": "associate", "holdings": "holding",
    "solutions": "solution", "industries": "industry", "systems": "system", "group": "group",
}

# Legal-form / filler tokens (after NAME_CANON). Removed to build the "core" name.
LEGAL = {"pvt", "ltd", "corp", "inc", "co", "llc", "llp", "pllc", "pc", "pa", "lp", "plc", "opc",
         "sarl", "sas", "sasu", "eurl", "sci", "snc", "sa", "cie", "gmbh", "&", "the", "l", "p", "c"}
FILLER = {"service", "group", "holding", "center", "ctr", "services", "company", "enterprise", "associate",
          "services", "partners", "solution", "fils"}
ALIAS_SPLIT = r"\b(?:dba|d b a|fka|f k a|aka|a k a|formerly|formerly known as|trading as|t a)\b"


def clean_expr(col):
    """Accent-stripped, lower-cased text with punctuation turned into spaces."""
    return (pl.col(col).fill_null("")
            .str.normalize("NFKD").str.replace_all(r"\p{Mn}", "")
            .str.to_lowercase()
            .str.replace_all(r"&", " & ")
            .str.replace_all(r"\.com\b|\.in\b|\.net\b|\.org\b|\.co\b|\.fr\b|www\.", " ")
            .str.replace_all(r"[^\p{L}\p{N}&]+", " ")
            .str.replace_all(r"\s+", " ").str.strip_chars())


def _multiword_states(expr):
    """Collapse multi-word state names into their codes before tokenising."""
    for full, code in sorted({**US_STATES, **IN_STATES}.items(), key=lambda kv: -len(kv[0])):
        if " " in full:
            expr = expr.str.replace_all(rf"\b{full}\b", code)
    return expr


def add_clean_columns(df):
    """Add `nm` (clean name) and `ad` (clean address) columns.

    Uses the transliterated `name_lat` / `addr_lat` columns when present.
    """
    name_col = "name_lat" if "name_lat" in df.columns else "business_name"
    addr_col = "addr_lat" if "addr_lat" in df.columns else "business_address"
    return df.with_columns(
        clean_expr(name_col).alias("nm"),
        _multiword_states(clean_expr(addr_col)).alias("ad"),
    )


def _map_tokens(tok_df, col, mapping):
    m = pl.DataFrame({col: list(mapping.keys()), "_to": list(mapping.values())})
    return (tok_df.join(m, on=col, how="left")
            .with_columns(pl.coalesce("_to", col).alias(col)).drop("_to"))


def _deleet(expr):
    """Undo digit-for-letter swaps inside alphabetic tokens (sh0shana -> shoshana)."""
    has_both = expr.str.contains(r"[a-z]") & expr.str.contains(r"[0-9]") & ~expr.str.contains(r"^[0-9]+[a-z]{1,2}$")
    fixed = (expr.str.replace_all("0", "o").str.replace_all("1", "l").str.replace_all("3", "e")
             .str.replace_all("5", "s").str.replace_all("4", "a").str.replace_all("7", "t"))
    return pl.when(has_both).then(fixed).otherwise(expr)


def name_tokens(df, id_col):
    """Long table (id, tok, pos) of canonical name tokens."""
    t = (df.select(pl.col(id_col), pl.col("nm").str.split(" ").alias("tok"))
           .explode("tok").filter(pl.col("tok") != "")
           .with_columns(_deleet(pl.col("tok")).alias("tok")))
    return _map_tokens(t, "tok", NAME_CANON)


def addr_tokens(df, id_col):
    """Long table (id, tok) of canonical address tokens."""
    t = (df.select(pl.col(id_col), pl.col("ad").str.split(" ").alias("tok"))
           .explode("tok").filter(pl.col("tok") != ""))
    t = _map_tokens(t, "tok", ADDR_CANON)
    return _map_tokens(t, "tok", {**US_STATES, **IN_STATES})
