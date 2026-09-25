"""Builds docs/Challenge_Overview.pdf — a readable brief of the challenge, data, rules and plan."""
import os
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, Preformatted)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Challenge_Overview.pdf")
FONTS = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("Arial", os.path.join(FONTS, "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", os.path.join(FONTS, "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", os.path.join(FONTS, "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Mono", os.path.join(FONTS, "consola.ttf")))
pdfmetrics.registerFont(TTFont("Sym", os.path.join(FONTS, "seguisym.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

INK = colors.HexColor("#1c2430")
MUTED = colors.HexColor("#5b6675")
ACCENT = colors.HexColor("#1f5f8b")
ACCENT_BG = colors.HexColor("#e8f0f7")
GOOD_BG = colors.HexColor("#e7f4ea")
GOOD = colors.HexColor("#1e6b35")
BAD_BG = colors.HexColor("#fbeaea")
BAD = colors.HexColor("#9b2226")
WARN_BG = colors.HexColor("#fff5e0")
RULE = colors.HexColor("#d5dbe3")

base = dict(fontName="Arial", textColor=INK)
S = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=22, leading=27, textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Arial", fontSize=11.5, leading=15, textColor=MUTED, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=14.5, leading=18, textColor=ACCENT, spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", fontSize=9.6, leading=13.4, spaceAfter=4, **base),
    "small": ParagraphStyle("small", fontSize=8.4, leading=11, **base),
    "cell": ParagraphStyle("cell", fontSize=8.6, leading=11.2, **base),
    "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.6, leading=11.2, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontSize=9.6, leading=13.2, leftIndent=12, bulletIndent=2, spaceAfter=1.5, **base),
    "callout": ParagraphStyle("callout", fontSize=9.8, leading=13.6, **base),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.2, leading=10.6, textColor=INK),
    "big": ParagraphStyle("big", fontName="Arial-Bold", fontSize=17, leading=20, textColor=ACCENT, alignment=TA_CENTER),
    "bigl": ParagraphStyle("bigl", fontSize=8.2, leading=10.5, textColor=MUTED, alignment=TA_CENTER),
}

P = lambda t, s="body": Paragraph(t, S[s])
C = lambda t: Paragraph(t, S["cell"])
CB = lambda t: Paragraph(t, S["cellb"])
W = A4[0] - 36 * mm


def bullets(items, style="bullet"):
    return [Paragraph(i, S[style], bulletText="•") for i in items]


def table(rows, widths, header=True, zebra=True, extra=()):
    data = [[CB(c) if (header and r == 0) else (c if not isinstance(c, str) else C(c)) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
          ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), ACCENT_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9fb")))
    t.setStyle(TableStyle(st + list(extra)))
    return t


def callout(text, bg=ACCENT_BG, bar=ACCENT):
    t = Table([[Paragraph(text, S["callout"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def code(text):
    t = Table([[Preformatted(text, S["code"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f5f8")),
                           ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def stat_row(stats):
    cells = [[Paragraph(v, S["big"]), Paragraph(l, S["bigl"])] for v, l in stats]
    inner = [Table([[a], [b]], colWidths=[W / len(stats) - 6]) for a, b in cells]
    for t in inner:
        t.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    t = Table([inner], colWidths=[W / len(stats)] * len(stats))
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, RULE), ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.white)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 7.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Amazon ML Challenge 2026 · Business Entity Resolution · Challenge overview")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.restoreState()


story = []
# ---------------------------------------------------------------- cover / summary
story += [P("Business Entity Resolution", "title"),
          P("Amazon ML Challenge 2026 — what the problem is, what we are given, what we must deliver, "
            "the rules we must not break, and how we plan to win. All data figures were measured on the "
            "provided files.", "subtitle")]
story.append(stat_row([("1.73M", "test Source-1 entities<br/>to resolve"), ("≈10M", "test Source-2 + Source-3<br/>records to search"),
                       ("F0.5", "macro, per entity<br/>(precision ×2)"), ("France", "15% of test, absent<br/>from training")]))
story.append(Spacer(1, 8))
story.append(P("1. The problem in plain words", "h1"))
story.append(P("Three independent systems each hold records about businesses. They share <b>no common ID</b>, and the "
               "same business is written differently in each one (typos, abbreviations, reordered addresses, "
               "names in Indian scripts, missing fields). <b>Source 1 is the clean, deduplicated master list.</b> "
               "For every Source 1 business in the test set, we must list every Source 2 and Source 3 record that is "
               "the same real-world business. A business can have zero, one, or many such records."))
story.append(code(
    "S1-496331216  Shree Best Business Private Limited\n"
    "              406, Manas Nagar Colony, Ankit Apartment, Lucknow, Uttar Pradesh\n"
    "   ├─ S2-198865808  <same name in Devanagari script>  | 406, MANAS NAGAR COLONY, ..., LUCKNOW\n"
    "   ├─ S3-43022101   Shree Best Business Limited Service | #851 406, UP, Lucknow\n"
    "   └─ S3-247018126  Shree Best Business Private-Ltd     | Door No 851 406, Lucknow, UP\n\n"
    "S1-299182413  East Advanced Cohen LLC | Pulaski, TN, 1189 Sumac Road\n"
    "   ├─ S2-957790650  East Advanced Cohen Llc | 1189 SUMAC RD, PLUASKI, TN\n"
    "   └─ S2-259918198  TAVOFLUX                | 1189 SUMAC ROAD, TN, PLUASKI   <- other trade name!"))
story.append(Spacer(1, 3))
story.append(P("<i>Real training examples: one S1 entity and its true S2/S3 matches.</i>", "small"))

# ---------------------------------------------------------------- inputs
story.append(P("2. What we are given (inputs)", "h1"))
story.append(P("All files are <b>UTF-8, tab-separated (.tsv)</b>. Each source file has four columns: "
               "<font name='Mono'>entity_id</font> (prefix S1-/S2-/S3- plus a random number), "
               "<font name='Mono'>business_name</font>, <font name='Mono'>business_address</font>, and "
               "<font name='Mono'>country</font>. The ground-truth file has "
               "<font name='Mono'>source1_entity_id</font> and <font name='Mono'>matched_entity_ids</font> "
               "(a comma-separated list, empty when there is no match)."))
story.append(table([
    ["File", "Rows", "Country mix", "Role"],
    ["train_source1.tsv", "2,206,821", "US 60% · India 40%", "Reference entities (labelled)"],
    ["train_source2.tsv", "5,034,616", "US 3.02M · India 2.02M", "Records to match"],
    ["train_source3.tsv", "5,285,603", "US 3.17M · India 2.12M", "Records to match"],
    ["train_ground_truth.tsv", "2,206,821", "—", "The answers for training"],
    ["test_source1.tsv", "1,732,544", "US 38% · India 47% · <b>France 15%</b>", "Entities we must answer for"],
    ["test_source2.tsv", "4,887,273", "US 1.87M · India 2.31M · France 0.70M", "Search pool"],
    ["test_source3.tsv", "5,082,316", "US 1.95M · India 2.41M · France 0.73M", "Search pool"],
], [W * 0.26, W * 0.13, W * 0.34, W * 0.27]))
story.append(Spacer(1, 4))
story.append(P("Also provided: <font name='Mono'>utils/validate_submission.py</font> (format checker), "
               "<font name='Mono'>Documentation_template.md</font> (methodology write-up to fill in), and a "
               "README identical to the problem statement. Total size is about 2.4 GB."))

story.append(P("3. What the data tells us (EDA)", "h1"))
story.append(table([
    ["Finding", "Evidence", "What it means for us"],
    ["Every S2/S3 record belongs to <b>at most one</b> S1", "7,638,365 links; every linked ID is used exactly once",
     "We can enforce a one-owner assignment, which is a strong precision tool."],
    ["It is one-to-many", "Mean 3.46 matches per S1; most have 2–6 (max 11)",
     "Returning only the single best match leaves recall on the table."],
    ["Singletons are rare", "5.6% of S1 have no match", "A wrong match on a singleton turns 1.0 into 0.0."],
    ["About a quarter of S2/S3 match nothing", "Unmatched: S2 26.6%, S3 25.4%", "Distractors are everywhere, so a similar name alone is not proof."],
    ["No cross-country matches", "0 of 7.6M links", "Block within each country label (lossless)."],
    ["Test pool is relatively larger", "S2+S3 per S1: train 4.68, test 5.75", "Expect a harder precision problem on test than on validation."],
    ["S1 is clean; S2/S3 are noisy", "S2/S3: ≈3% empty addresses; ≈40% of Indian records in Indic scripts (9 scripts)",
     "We need normalisation and transliteration."],
    ["Postal codes are basically absent", "India PIN: 0; US ZIP: under 0.01%", "We can't block on ZIP/PIN; use city, street and house number."],
    ["IDs carry no information", "No train/test ID overlap; random numbers", "Never use ID values as features."],
    ["Parsing traps", "≈800 test rows with quoted fields like \"\"\"ehpad Club SAS\"; ≈100 NA/null values",
     "Load with quoting off and NA-conversion off, then assert row counts."],
], [W * 0.27, W * 0.36, W * 0.37]))
story.append(P("Noise patterns seen in true matches", "h2"))
story += bullets([
    "<b>Names:</b> typos, leetspeak (<i>Sh0shana, 5hoshana, Carg0</i>), random accents (<i>Émpire</i>), legal suffix added, dropped or "
    "reworded (<i>Pvt Ltd / Private Limited</i>), filler words (<i>Center, Services, Group</i>), word reordering, junk "
    "prefixes (<i>--, &lt;&lt;, #</i>), domains and handles (<i>pediatricmedicine.com</i>), aliases (<i>X dba Y, X fka Y</i>), a "
    "completely different trade name at the same address, and whole names transliterated into Indic scripts.",
    "<b>Addresses:</b> components reordered, <i>Rd/Road</i>, <i>TN/Tennessee</i>, <i>UP/Uttar Pradesh</i>, house number "
    "off by a little (<i>2046/2048</i>) or given as a range (<i>120-122</i>), injected <i>#, Door No, HN, PO Box</i>, "
    "neighbouring or renamed cities (<i>Gurgaon/Gurugram</i>), city typos, missing parts, or no address at all.",
    "<b>France (test only):</b> SARL/SAS/SASU/EURL/SCI legal forms, <i>R./AV./BD.</i> street abbreviations, region vs "
    "département (<i>Hauts-de-France / Nord</i>), accents on and off, few cities, and very generic names "
    "(<i>Association des Coeur, Clinique Jeanne</i>).",
])

# ---------------------------------------------------------------- outputs
story.append(P("4. What we must produce (outputs)", "h1"))
story.append(P("<b>(a) output/matching_results.tsv</b> — the only scored file, uploaded to the leaderboard portal."))
story.append(code("source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00047,S2-00193,S3-00812\nS1-00002\tS3-00004\nS1-00003\t\n"
                  "                                   ^ empty = we believe this entity has no match"))
story.append(Spacer(1, 4))
story.append(P("<b>(b) output/candidate_pairs.tsv</b> — not scored, but audited. It must be the <b>exact</b> candidate list that our "
               "matching model scores (the last blocking/filter stage), with header "
               "<font name='Mono'>source1_entity_id &lt;TAB&gt; candidate_entity_ids</font>. Every matched ID must also appear here. The "
               "organisers use it to measure our blocking recall and reduction ratio."))
story.append(P("<b>(c) The final zip</b> — <font name='Mono'>&lt;team_name&gt;_submission.zip</font>:"))
story.append(code("<team_name>_submission.zip\n├── output/\n│   ├── matching_results.tsv\n│   └── candidate_pairs.tsv\n"
                  "├── code/business_entity_resolution/\n│   ├── src/               all source code\n"
                  "│   ├── README.md          exact steps: data → blocking → matching → output\n"
                  "│   └── requirements.txt   pinned versions\n└── Documentation_template.md   filled-in methodology (.md or .pdf)"))
story.append(Spacer(1, 4))
story.append(P("The methodology document must cover the method, the blocking strategy, the model and features, and results "
               "with error analysis. There is no page limit, and depth is valued."))

# ---------------------------------------------------------------- scoring
story.append(P("5. How we are scored", "h1"))
story.append(callout("<b>F0.5 = 1.25 · P · R / (0.25 · P + R)</b>, computed <b>for each Source 1 entity</b> and then "
                     "<b>averaged over all of them</b> (macro average). Precision counts twice as much as recall. "
                     "Truth empty and prediction empty = <b>1.0</b>. Truth empty and anything predicted = <b>0.0</b>. "
                     "Truth non-empty and prediction empty = <b>0.0</b>."))
story.append(Spacer(1, 5))
story.append(table([
    ["Situation (true matches = k)", "Prediction", "P", "R", "F0.5"],
    ["Problem-statement example, k = 2", "2 right + 1 wrong", "0.67", "1.00", "0.714"],
    ["k = 3", "2 right, 0 wrong", "1.00", "0.67", "<b>0.909</b>"],
    ["k = 3", "3 right + 1 wrong", "0.75", "1.00", "0.789"],
    ["k = 5", "only the 1 surest match", "1.00", "0.20", "0.556"],
    ["Singleton, k = 0", "empty", "—", "—", "<b>1.000</b>"],
    ["Singleton, k = 0", "any 1 match", "0", "—", "0.000"],
], [W * 0.32, W * 0.28, W * 0.12, W * 0.12, W * 0.16]))
story.append(Spacer(1, 4))
story.append(P("<b>Takeaway:</b> a missing link costs less than a wrong link, but an empty answer for a real entity scores zero. "
               "The best strategy is to add matches in order of confidence and stop as soon as the next one is more "
               "likely to hurt than help. The public leaderboard uses only a subset of the test set. <b>Final ranking uses the "
               "private remainder</b>, so thresholds must not be tuned to the public score."))

# ---------------------------------------------------------------- constraints
story.append(P("6. Constraints and rules", "h1"))
story.append(table([
    ["Area", "Rule", "Consequence if broken"],
    ["Format", "Tab-separated, exact header names, UTF-8; one row per test S1; only existing S2-/S3- IDs; no duplicate IDs in a list; no duplicate rows",
     "Submission rejected (not scored)"],
    ["Completeness", "Every test S1 must appear, including France and singletons (empty field)", "Rejected"],
    ["Model", "Final model must be <b>MIT or Apache-2.0</b> licensed and <b>≤ 8 billion parameters</b>", "Fails license review"],
    ["Fair play", "<b>No external data:</b> no entity-resolution APIs, business registries, <b>geocoding APIs</b>, or internet augmentation",
     "<b>Immediate disqualification</b>"],
    ["Reproducibility", "Zip must regenerate both outputs from the raw data with only the shipped code, README and requirements",
     "Top teams audited; may be disqualified"],
    ["Country", "Country is an open set: no hard-coding, filtering or one-hot to {US, India}", "France rows lost or mis-scored"],
], [W * 0.16, W * 0.56, W * 0.28],
    extra=[("BACKGROUND", (2, 4), (2, 4), BAD_BG)]))
story.append(Spacer(1, 5))
story.append(P("<b>License quick-guide</b> (check each model card before use). OK: LightGBM (MIT), XGBoost and CatBoost (Apache-2.0), "
               "multilingual MiniLM and LaBSE (Apache-2.0), multilingual-e5 and bge-m3 (MIT), Qwen2.5-7B and Mistral-7B (Apache-2.0). "
               "<b>Not OK:</b> Llama, Gemma, any model over 8B, non-commercial licenses. Keep libraries permissive too "
               "(e.g. avoid GPL Unidecode)."))
story.append(P("<b>Our hardware:</b> i5-13500H (16 threads), 15.6 GB RAM (often under 1 GB free), no CUDA GPU, about 12 GB free disk. "
               "The current Python environment is missing rapidfuzz, lightgbm and polars, and torch is broken against numpy 2, so "
               "we need a clean virtual environment. Work one country at a time and in chunks, and never build the full cross product "
               "(India alone is about 0.8M × 4.7M pairs)."))

# ---------------------------------------------------------------- do / don't
dd_head = P("7. Do and don't", "h1")
dos = [
    "Read with <font name='Mono'>sep='\\t', dtype=str, quoting=QUOTE_NONE, keep_default_na=False</font> and assert row counts",
    "Write with <font name='Mono'>sep='\\t', index=False, lineterminator='\\n'</font> (Windows defaults to CRLF)",
    "Group by whatever country strings exist; never enumerate them",
    "Put every test S1 in both files, including France and singletons",
    "Enforce one owner per S2/S3 record",
    "Tune thresholds for macro per-entity F0.5 on out-of-fold data",
    "Keep candidate_pairs.tsv equal to the model's real inference set",
    "Run the validator before every upload (sometimes with --check-ids)",
    "Pin versions, fix seeds, use relative paths",
    "Disclose all rule tables, libraries and model licenses in the documentation",
]
donts = [
    "Call any external API or geocoder, or download business or address data",
    "Hard-code, one-hot or filter countries to {US, India}",
    "Use Llama/Gemma or any model over 8B in the pipeline",
    "Force a \"best available\" match: singletons need an empty answer",
    "Chase the public leaderboard (it is only a subset)",
    "Rely on ZIP or PIN codes (they are essentially absent)",
    "Trust train-validation scores for France without a margin",
    "Write CSV, quoted fields, CRLF, or spaces after commas",
    "Use entity ID numbers as a signal",
    "Let pandas turn 'NA' names into NaN or merge quoted rows",
]
rows = [[Paragraph("<font color='#1e6b35'><b>DO</b></font>", S["cellb"]), Paragraph("<font color='#9b2226'><b>DON'T</b></font>", S["cellb"])]]
for a, b in zip(dos, donts):
    rows.append([Paragraph("<font name='Sym' color='#1e6b35'>✓</font>&nbsp; " + a, S["cell"]), Paragraph("<font name='Sym' color='#9b2226'>✗</font>&nbsp; " + b, S["cell"])])
dt = Table(rows, colWidths=[W / 2, W / 2])
dt.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), GOOD_BG), ("BACKGROUND", (1, 0), (1, -1), BAD_BG),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.white),
                        ("LINEAFTER", (0, 0), (0, -1), 2, colors.white),
                        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
story.append(KeepTogether([dd_head, dt]))

# ---------------------------------------------------------------- plan
story.append(P("8. How we will solve it", "h1"))
story.append(code(
    "raw TSVs\n"
    " → [0] robust load + integrity asserts\n"
    " → [1] normalise: text clean, leetspeak fix, alias split (dba/fka/aka), legal-form extraction,\n"
    "        domain → name, Indic → Latin transliteration, address parse (house no., street, city, state)\n"
    " → [2] blocking per country label: each S2/S3 record retrieves its top-K S1 candidates via\n"
    "        rare-token keys + char-3gram TF-IDF kNN + address keys + concat-name key + embedding kNN\n"
    "        ==> candidate_pairs.tsv\n"
    " → [3] pairwise features (name, address, rarity, rank/margin; country NOT used as a feature)\n"
    " → [4] LightGBM matcher  (+ optional ≤8B MIT/Apache cross-encoder on the uncertain band)\n"
    " → [5] decision: one owner per S2/S3, runner-up margin, keep the prefix that maximises expected F0.5\n"
    "        ==> matching_results.tsv → validator → upload"))
story.append(Spacer(1, 5))
story.append(P("<b>Validation:</b> 5-fold GroupKFold by S1 over training, with blocking against the full S2/S3 pool so distractors are "
               "realistic. Metric re-implemented exactly (unit-tested on the 0.714 example). <b>Leave-one-country-out</b> "
               "(train US → test India, and the reverse) serves as a stand-in for France, since no French labels exist. "
               "Track the blocking recall ceiling (target ≥ 98% of true links)."))
story.append(KeepTogether([P("Milestones", "h2"), table([
    ["#", "Milestone", "Done when"],
    ["M0", "Environment, loader, metric, writer", "Clean venv; row-count asserts; metric tests; an all-empty file passes the validator"],
    ["M1", "Rule baseline on the leaderboard", "First SCORED submission; baseline logged"],
    ["M2", "Multi-pass blocking", "≥ 98% pair recall on validation within RAM; candidate_pairs.tsv produced"],
    ["M3", "ML matcher (LightGBM)", "Out-of-fold macro F0.5 clearly above M1"],
    ["M4", "Multilingual and France robustness", "Learned transliteration; French rules; leave-one-country-out gap reduced"],
    ["M5", "Decision layer", "Assignment + expected-F0.5 selection tuned out-of-fold"],
    ["M6", "Package", "Zip reproduces end-to-end from a clean copy; docs filled; validator PASS"],
], [W * 0.07, W * 0.33, W * 0.60])]))

# ---------------------------------------------------------------- checklist + open questions
story.append(KeepTogether([
    P("9. Pre-submission checklist", "h1"),
    *bullets([
        "<font name='Sym'>☐</font> 1,732,544 data rows plus header in both output files; header names exact",
        "<font name='Sym'>☐</font> Tabs only, UTF-8, LF line endings, no quotes, no spaces in ID lists",
        "<font name='Sym'>☐</font> Only S2-/S3- IDs that exist in the test files; no duplicates; every match is also a candidate",
        "<font name='Sym'>☐</font> <font name='Mono'>python utils/validate_submission.py --matching output/matching_results.tsv "
        "--candidate output/candidate_pairs.tsv --test-dir dataset/test</font> prints PASS",
        "<font name='Sym'>☐</font> Zip layout exactly as specified; README reproduces from scratch; requirements pinned",
        "<font name='Sym'>☐</font> Documentation_template.md filled in; licenses and rule tables disclosed; no external data used",
    ]),
]))
story.append(P("10. Open questions", "h1"))
story += bullets([
    "Deadline and daily leaderboard submission limit (not stated in the materials).",
    "Team name and member list (needed for the zip and the documentation).",
    "Confirm hand-written normalisation tables and offline transliteration are acceptable (we will disclose them either way).",
    "Confirm unsupervised use of test text (e.g. IDF fitting) is acceptable.",
    "Does the MIT/Apache rule apply to every dependency? (We assume yes, to be safe.)",
    "Watch the linked explainer video; it is not included in the files and may add details.",
])
story.append(Spacer(1, 6))
story.append(P("Full detail: docs/PRD.md. Raw EDA numbers: docs/eda_stats.json.", "small"))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                        bottomMargin=18 * mm, title="Business Entity Resolution — Challenge Overview",
                        author="ML Challenge team", subject="Amazon ML Challenge 2026 brief")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
