import re

# =====================================================
#  CASE METADATA REGEXES
# =====================================================

# --- Case title: e.g. "Aamir vs State of U.P."
case_title_re = re.compile(
    r"(?P<p1>[A-Z][\w\s\.\&]+)\s+v(?:s\.?|ersus)\s+(?P<p2>[A-Z][\w\s\.\&]+)",
    re.I
)

# --- Court names ---
court_re = re.compile(
    r"(SUPREME COURT OF INDIA|HIGH COURT OF [A-Z\s]+|DISTRICT COURT OF [A-Z\s]+|COURT OF THE SESSIONS JUDGE[, ]+[A-Z\s]+)",
    re.I
)

# --- IPC / Section extraction ---
section_re = re.compile(r"(?:IPC\s*)?(?:S\.|Sec(?:tion)?)\s*\.?\s*(\d{1,3}[A-Z]?)", re.I)

# --- Acts (common examples, expandable) ---
import re

# This is a comprehensive list combining major central acts and key Uttar Pradesh state acts.
act_patterns = [
    # -- Core Procedural & Penal Codes --
    re.compile(r"Indian Penal Code|IPC", re.I),
    re.compile(r"Code of Criminal Procedure|CrPC", re.I),
    re.compile(r"Code of Civil Procedure|CPC", re.I),
    re.compile(r"Indian Evidence Act|Evidence Act", re.I),

    # -- Major Criminal Acts (Central) --
    re.compile(r"Narcotic Drugs and Psychotropic Substances Act|NDPS Act", re.I),
    re.compile(r"Protection of Children from Sexual Offences Act|POCSO Act", re.I),
    re.compile(r"Prevention of Corruption Act|PC Act", re.I),
    re.compile(r"Unlawful Activities \(Prevention\) Act|UAPA", re.I),
    re.compile(r"Prevention of Money Laundering Act|PMLA", re.I),
    re.compile(r"Probation of Offenders Act", re.I),
    
    # -- Family & Personal Law (Central) --
    re.compile(r"Dowry Prohibition Act", re.I),
    re.compile(r"Protection of Women from Domestic Violence Act|PWDVA|DV Act", re.I),
    re.compile(r"Hindu Marriage Act", re.I),
    re.compile(r"Hindu Succession Act", re.I),
    re.compile(r"Special Marriage Act", re.I),
    re.compile(r"Guardians and Wards Act", re.I),
    re.compile(r"Juvenile Justice \(Care and Protection of Children\) Act|JJ Act", re.I),

    # -- Core Civil, Contract & Property Law (Central) --
    re.compile(r"Indian Contract Act", re.I),
    re.compile(r"Transfer of Property Act", re.I),
    re.compile(r"Specific Relief Act", re.I),
    re.compile(r"Limitation Act", re.I),

    # -- Corporate, Commercial & Financial Law (Central) --
    re.compile(r"Companies Act", re.I),
    re.compile(r"Negotiable Instruments Act|NI Act", re.I),
    re.compile(r"Arbitration and Conciliation Act", re.I),
    re.compile(r"Insolvency and Bankruptcy Code|IBC", re.I),
    re.compile(r"Securities and Exchange Board of India Act|SEBI Act", re.I),
    re.compile(r"Banking Regulation Act", re.I),

    # -- Administrative & Regulatory Acts (Central) --
    re.compile(r"Information Technology Act|IT Act", re.I),
    re.compile(r"Right to Information Act|RTI Act", re.I),
    re.compile(r"Consumer Protection Act", re.I),
    re.compile(r"Environment \(Protection\) Act", re.I),
    re.compile(r"Motor Vehicles Act|MV Act", re.I),
    re.compile(r"Arms Act", re.I),
    re.compile(r"Essential Commodities Act", re.I),
    re.compile(r"Foreigners Act", re.I),
    
    # -- Social Justice & Labor Law (Central) --
    re.compile(r"SC/ST \(Prevention of Atrocities\) Act|SC/ST Act", re.I),
    re.compile(r"Industrial Disputes Act", re.I),
    re.compile(r"Factories Act", re.I),
    re.compile(r"Minimum Wages Act", re.I),

    # -- Tax Law (Central) --
    re.compile(r"Income Tax Act", re.I),
    re.compile(r"Central Goods and Services Tax Act|CGST Act", re.I),
    re.compile(r"Integrated Goods and Services Tax Act|IGST Act", re.I),
    
    # -- Uttar Pradesh (U.P.) Specific Acts --
    re.compile(r"U\.P\. Gangsters and Anti-Social Activities \(Prevention\) Act|Gangsters Act", re.I),
    re.compile(r"U\.P\. Revenue Code", re.I),
    re.compile(r"U\.P\. Zamindari Abolition and Land Reforms Act", re.I),
    re.compile(r"U\.P\. Panchayat Raj Act", re.I),
    re.compile(r"U\.P\. Municipalities Act", re.I),
    re.compile(r"U\.P\. Excise Act", re.I),
    re.compile(r"U\.P\. Urban Buildings \(Regulation of Letting, Rent and Eviction\) Act|U\.P\. Rent Control Act", re.I),
    re.compile(r"U\.P\. Industrial Area Development Act", re.I),

    # -- Special case for the Constitution --
    re.compile(r"Constitution of India|Constitution", re.I),
]

# A separate pattern for Articles remains useful
article_pattern = re.compile(r"Article\s+\d+", re.I)

# =====================================================
#  DATE REGEXES — Handles multiple Indian + ISO formats
# =====================================================

# Example: 31/01/2025 or 31-01-2025
date_re_slash = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")

# Example: 31 January 2025 or 31 January, 2025
date_re_textual = re.compile(
    r"\b(\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s*\d{4})\b",
    re.IGNORECASE
)

# Example: 2025-01-31 (ISO)
date_re_iso = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# Example: 31.01.2025 (dot separated)
date_re_dot = re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b")

# Combined fallback (for robustness)
date_re_any = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s*\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\.\d{1,2}\.\d{2,4})\b",
    re.IGNORECASE,
)

# =====================================================
#  PLACE REGEXES — Augments indian_places.json lookups
# =====================================================

# High-level State / UT name pattern (used before JSON lookup)
state_re = re.compile(
    r"\b(Andhra Pradesh|Arunachal Pradesh|Assam|Bihar|Chhattisgarh|Goa|Gujarat|Haryana|Himachal Pradesh|Jharkhand|Karnataka|Kerala|Madhya Pradesh|Maharashtra|Manipur|Meghalaya|Mizoram|Nagaland|Odisha|Punjab|Rajasthan|Sikkim|Tamil Nadu|Telangana|Tripura|Uttar Pradesh|Uttarakhand|West Bengal|Delhi|Jammu and Kashmir|Ladakh|Puducherry|Chandigarh|Andaman and Nicobar Islands|Lakshadweep)\b",
    re.IGNORECASE
)

# Common city/district indicators for extra hinting
place_hint_re = re.compile(
    r"\b(in|at|from|near|district|city|village|town|tehsil|p.s.|p\.s|police station|court of|session at)\s+([A-Z][a-z]+)\b",
    re.IGNORECASE
)
