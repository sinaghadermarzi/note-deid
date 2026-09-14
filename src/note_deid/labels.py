"""Label taxonomies and cross-model mapping tables.

Canonical label set: the i2b2/UTHealth 2014 scheme — 7 categories, 28 leaf labels (25 sub-categories under
NAME/LOCATION/CONTACT/ID plus the undivided PROFESSION, AGE, DATE). Every corpus loader
maps its native labels to this set; every model backend maps to/from it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# -- i2b2 2014 taxonomy (Stubbs & Uzuner 2015) -------------------------------------------------------------------
I2B2_2014_CATEGORIES: dict[str, tuple[str, ...]] = {
    "NAME": ("PATIENT", "DOCTOR", "USERNAME"),
    "PROFESSION": ("PROFESSION",),
    "LOCATION": ("HOSPITAL", "ORGANIZATION", "STREET", "CITY", "STATE", "COUNTRY", "ZIP", "LOCATION-OTHER"),
    "AGE": ("AGE",),
    "DATE": ("DATE",),
    "CONTACT": ("PHONE", "FAX", "EMAIL", "URL", "IPADDR"),
    "ID": ("SSN", "MEDICALRECORD", "HEALTHPLAN", "ACCOUNT", "LICENSE", "VEHICLE", "DEVICE", "BIOID", "IDNUM"),
}
I2B2_2014_SUBTYPES: tuple[str, ...] = tuple(s for subs in I2B2_2014_CATEGORIES.values() for s in subs)
SUBTYPE_TO_CATEGORY: dict[str, str] = {s: c for c, subs in I2B2_2014_CATEGORIES.items() for s in subs}

# Optional extension for institution-specific identifiers (framework-designs.md §1.1); off by default.
INSTITUTIONAL_SUBTYPES: tuple[str, ...] = ("INSTITUTIONAL",)

# Subtypes counted in the i2b2-2014 "HIPAA-only" view (Stubbs et al. 2015; verify against the corpus paper table).
# AGE is Safe-Harbor PHI only when >= 90 and a bare year is not a Safe-Harbor date element; the evaluator applies
# both rules when computing the HIPAA view.
HIPAA_SUBTYPES: frozenset[str] = frozenset(
    {
        "PATIENT",
        "USERNAME",
        "STREET",
        "CITY",
        "ZIP",
        "LOCATION-OTHER",
        "AGE",
        "DATE",
        "PHONE",
        "FAX",
        "EMAIL",
        "URL",
        "IPADDR",
        "SSN",
        "MEDICALRECORD",
        "HEALTHPLAN",
        "ACCOUNT",
        "LICENSE",
        "VEHICLE",
        "DEVICE",
        "BIOID",
        "IDNUM",
    }
)

HIPAA_SAFE_HARBOR_IDENTIFIERS: tuple[str, ...] = (
    "names",
    "geographic subdivisions smaller than a state",
    "dates (except year) directly related to an individual; ages over 89",
    "telephone numbers",
    "fax numbers",
    "email addresses",
    "social security numbers",
    "medical record numbers",
    "health plan beneficiary numbers",
    "account numbers",
    "certificate/license numbers",
    "vehicle identifiers and serial numbers",
    "device identifiers and serial numbers",
    "URLs",
    "IP addresses",
    "biometric identifiers",
    "full-face photographs and comparable images",
    "any other unique identifying number, characteristic, or code",
)

# -- OpenAI Privacy Filter (8 labels x BIOES = 33 classes) --------------------------------------------------------
OPF_LABELS: tuple[str, ...] = (
    "private_person",
    "private_date",
    "private_address",
    "private_phone",
    "private_email",
    "private_url",
    "account_number",
    "secret",
)

# Canonical subtype -> OPF label. ``None`` = OPF has no label for it (the "unsupported" set used by H2/H3/H13).
I2B2_TO_OPF: dict[str, str | None] = {
    "PATIENT": "private_person",
    "DOCTOR": "private_person",
    "USERNAME": None,
    "PROFESSION": None,
    "HOSPITAL": None,
    "ORGANIZATION": None,
    "STREET": "private_address",
    "CITY": "private_address",
    "STATE": "private_address",
    "COUNTRY": None,
    "ZIP": "private_address",
    "LOCATION-OTHER": "private_address",
    "AGE": None,
    "DATE": "private_date",
    "PHONE": "private_phone",
    "FAX": "private_phone",
    "EMAIL": "private_email",
    "URL": "private_url",
    "IPADDR": None,
    "SSN": "account_number",
    "MEDICALRECORD": "account_number",
    "HEALTHPLAN": "account_number",
    "ACCOUNT": "account_number",
    "LICENSE": "account_number",
    "VEHICLE": "account_number",
    "DEVICE": "account_number",
    "BIOID": None,
    "IDNUM": "account_number",
}
OPF_UNSUPPORTED_SUBTYPES: frozenset[str] = frozenset(k for k, v in I2B2_TO_OPF.items() if v is None)

# OPF label -> canonical *category* (subtype is not recoverable from OPF output).
OPF_TO_I2B2_CATEGORY: dict[str, str | None] = {
    "private_person": "NAME",
    "private_date": "DATE",
    "private_address": "LOCATION",
    "private_phone": "CONTACT",
    "private_email": "CONTACT",
    "private_url": "CONTACT",
    "account_number": "ID",
    "secret": None,
}

# -- OpenMed-PII (54 labels; DeBERTa-v3-large, trained on Nemotron-PII) ------------------------------------------
# Keyed by *normalized* label name (see ``normalize_label``) because the checkpoint's id2label strings vary in
# case/punctuation. Unlisted labels resolve to ``None`` (not PHI under our taxonomy) unless ``strict`` is requested.
OPENMED_TO_I2B2: dict[str, str | None] = {
    # identifiers
    "accountnumber": "ACCOUNT",
    "apikey": None,
    "bankroutingnumber": "IDNUM",
    "certificatelicensenumber": "LICENSE",
    "licensenumber": "LICENSE",
    "creditdebitcard": "ACCOUNT",
    "creditcard": "ACCOUNT",
    "cvv": None,
    "employeeid": "IDNUM",
    "healthplanbeneficiarynumber": "HEALTHPLAN",
    "macaddress": "DEVICE",
    "medicalrecordnumber": "MEDICALRECORD",
    "passportnumber": "IDNUM",
    "phonenumber": "PHONE",
    "pin": None,
    "ssn": "SSN",
    "swiftcode": "IDNUM",
    "vin": "VEHICLE",
    # personal info
    "age": "AGE",
    "biometricidentifier": "BIOID",
    "bloodtype": None,
    "dateofbirth": "DATE",
    "educationlevel": None,
    "firstname": "PATIENT",
    "lastname": "PATIENT",
    "name": "PATIENT",
    "gender": None,
    "language": None,
    "maritalstatus": None,
    "nationality": None,
    "occupation": "PROFESSION",
    "raceethnicity": None,
    "religion": None,
    "sexuality": None,
    # contact
    "email": "EMAIL",
    "faxnumber": "FAX",
    "url": "URL",
    # location
    "city": "CITY",
    "coordinate": "LOCATION-OTHER",
    "country": "COUNTRY",
    "county": "LOCATION-OTHER",
    "state": "STATE",
    "streetaddress": "STREET",
    # network
    "deviceidentifier": "DEVICE",
    "ipv4": "IPADDR",
    "ipv6": "IPADDR",
    "ipaddress": "IPADDR",
    # temporal
    "date": "DATE",
    "datetime": "DATE",
    "time": None,
    # organization
    "companyname": "ORGANIZATION",
    "organization": "ORGANIZATION",
    "hospital": "HOSPITAL",
}

# -- helpers -----------------------------------------------------------------------------------------------------
_BIO_PREFIX = re.compile(r"^[BIESLU]-", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]")


def strip_bio_prefix(tag: str) -> str:
    """'B-PATIENT' -> 'PATIENT'; 'b-patient' -> 'patient'; 'O' -> 'O'. Case-insensitive on the prefix."""
    return _BIO_PREFIX.sub("", tag)


def normalize_label(name: str, strip_prefix: bool = True) -> str:
    """Lowercase and drop all non-alphanumerics, after removing a BIO/BIOES prefix unless ``strip_prefix=False``.

    'B-Medical_Record-Number' -> 'medicalrecordnumber'; 'b-medical_record_number' -> 'medicalrecordnumber'
    """
    base = strip_bio_prefix(name) if strip_prefix else name
    return _NON_ALNUM.sub("", base.lower())


def map_label(label: str, table: Mapping[str, str | None], strict: bool = False) -> str | None:
    """Map a model/corpus label through ``table`` (exact key first, then normalized key).

    Returns ``None`` when the table says the label is not PHI under the canonical taxonomy. With ``strict=True`` an
    unknown label raises ``KeyError`` instead of returning ``None``.
    """
    if label in table:
        return table[label]
    # Normalized key with the BIO prefix stripped, then without stripping so that a label whose own name starts
    # like a prefix (e.g. 'E-mail') is not mangled.
    for key in (normalize_label(label), normalize_label(label, strip_prefix=False)):
        if key in table:
            return table[key]
    if strict:
        raise KeyError(label)
    return None


def bioes_tags(labels: tuple[str, ...] | list[str]) -> list[str]:
    """Token-classification class list for a label set: ['O', 'B-x', 'I-x', 'E-x', 'S-x', ...]."""
    return ["O"] + [f"{p}-{lab}" for lab in labels for p in ("B", "I", "E", "S")]


# Labels a checkpoint may emit without any mapping: the 28 subtypes, the 7 categories, the binary label and the
# optional institutional extension (the label spaces of ``note_deid.tc.train``).
CANONICAL_MODEL_LABELS: frozenset[str] = (
    frozenset(I2B2_2014_SUBTYPES) | frozenset(I2B2_2014_CATEGORIES) | frozenset(INSTITUTIONAL_SUBTYPES) | {"PHI"}
)


def infer_label_map(model_labels: Iterable[str]) -> str | None:
    """Name of the mapping a checkpoint's labels need before evaluation: ``None`` when they are canonical,
    ``"opf_category"`` for the native OpenAI Privacy Filter labels, ``"openmed"`` for OpenMed-PII.

    BIO/BIOES prefixes and ``O`` are ignored. Raises ``ValueError`` for an empty, mixed or unknown label set so that
    raw model labels never reach the evaluator silently.
    """
    labels = {strip_bio_prefix(x) for x in model_labels if x != "O"}
    if not labels:
        raise ValueError("checkpoint has no entity labels")
    if labels <= CANONICAL_MODEL_LABELS:
        return None
    if labels <= set(OPF_LABELS):
        return "opf_category"
    known = sum(1 for x in labels if normalize_label(x) in OPENMED_TO_I2B2)
    if known * 2 >= len(labels):
        return "openmed"
    unknown = sorted(labels - CANONICAL_MODEL_LABELS - set(OPF_LABELS))
    raise ValueError(
        f"cannot infer a label mapping for model labels {unknown}; pass label_map explicitly "
        "(opf_category | openmed) or train with a canonical label space"
    )


# Training-frequency buckets used in every per-label analysis (framework-designs.md §2).
FREQUENCY_BUCKETS: tuple[tuple[int, int | None], ...] = ((0, 0), (1, 10), (11, 50), (51, 200), (201, None))


def frequency_bucket(n: int) -> str:
    """'0', '1-10', '11-50', '51-200', '>200' for a per-label training count ``n``."""
    for lo, hi in FREQUENCY_BUCKETS:
        if hi is None:
            if n >= lo:
                return f">{lo - 1}"
        elif lo <= n <= hi:
            return f"{lo}-{hi}" if lo != hi else str(lo)
    raise ValueError(f"negative count: {n}")
