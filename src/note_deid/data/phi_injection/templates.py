"""Slot templates (headers, signature blocks) and per-subtype sentence banks. Placeholders are ``{LABEL}`` or
``{LABEL:form}``; rendering returns text plus gold spans by construction."""

from __future__ import annotations

import re
from collections.abc import Callable

from note_deid.schema import Span

PLACEHOLDER = re.compile(r"\{([A-Z][A-Z-]*)(?::([a-z_]+))?\}")

HEADERS: list[str] = [
    "PATIENT: {PATIENT:full}\nMRN: {MEDICALRECORD}\nDOB: {DATE:dob}\nDATE OF SERVICE: {DATE:visit}\n"
    "ATTENDING: {DOCTOR:full}\nFACILITY: {HOSPITAL}",
    "{HOSPITAL}\n{STREET}, {CITY}, {STATE} {ZIP}\nPhone {PHONE}\n\nName: {PATIENT:full}    MRN: {MEDICALRECORD}    "
    "DOB: {DATE:dob}\nVisit date: {DATE:visit}    Provider: {DOCTOR}",
    "Patient Name: {PATIENT:full}\nDate of Birth: {DATE:dob}\nMedical Record #: {MEDICALRECORD}\nAdmission Date: "
    "{DATE:visit}\nDischarge Date: {DATE:discharge}\nAttending Physician: {DOCTOR:full}",
    "RE: {PATIENT:full} (MRN {MEDICALRECORD})\nDate: {DATE:visit}\nTo: {DOCTOR:full}, {HOSPITAL}",
]

SIGNATURES: list[str] = [
    "Electronically signed by {DOCTOR:full} on {DATE:discharge}.",
    "Dictated by: {DOCTOR:full}\nTranscribed: {DATE:discharge}\ncc: {DOCTOR:full}",
    "{DOCTOR:full}\n{HOSPITAL}\nTel {PHONE} | Fax {FAX}",
    "Signed: {DOCTOR}, {DATE:discharge}\nQuestions: {EMAIL}",
]

SENTENCES: dict[str, list[str]] = {
    "PATIENT": [
        "{PATIENT} was accompanied by family.",
        "Discussed the findings with {PATIENT} at length.",
        "{PATIENT} verbalized understanding of the plan.",
    ],
    "DOCTOR": [
        "The case was discussed with {DOCTOR}.",
        "{DOCTOR} will follow the patient in clinic.",
        "Referred to {DOCTOR} for further evaluation.",
    ],
    "USERNAME": ["Portal account {USERNAME} was activated for result release.", "Documented under user {USERNAME}."],
    "PROFESSION": ["The patient works as a {PROFESSION}.", "Occupation: {PROFESSION}."],
    "HOSPITAL": ["Records were requested from {HOSPITAL}.", "Previously admitted to {HOSPITAL}."],
    "ORGANIZATION": ["The patient is employed by {ORGANIZATION}.", "Insurance is through {ORGANIZATION}."],
    "STREET": ["Home address: {STREET}, {CITY}, {STATE} {ZIP}.", "Lives at {STREET}."],
    "CITY": ["The patient recently moved to {CITY}.", "Lives in {CITY} with a spouse."],
    "STATE": ["Relocated from {STATE} last year.", "Prior care was in {STATE}."],
    "COUNTRY": ["Returned from travel to {COUNTRY} two weeks ago.", "Born in {COUNTRY}."],
    "ZIP": ["Mailing address zip code {ZIP}.", "Pharmacy in the {ZIP} area."],
    "LOCATION-OTHER": ["The fall occurred near {LOCATION-OTHER}.", "Walks daily at {LOCATION-OTHER}."],
    "AGE": ["The patient is a {AGE}-year-old.", "Age {AGE}.", "This {AGE} year old presents for follow-up."],
    "DATE": [
        "Seen on {DATE:visit}.",
        "Symptoms began on {DATE:visit}.",
        "Follow-up scheduled for {DATE:followup}.",
        "Last visit was in {DATE:partial}.",
    ],
    "PHONE": ["The patient can be reached at {PHONE}.", "Contact number: {PHONE}."],
    "FAX": ["Please fax results to {FAX}.", "Fax: {FAX}."],
    "EMAIL": ["Email on file: {EMAIL}.", "Results will be sent to {EMAIL}."],
    "URL": ["Patient education materials at {URL}.", "See {URL} for the home exercise program."],
    "IPADDR": ["Remote monitoring device reporting from {IPADDR}.", "Telehealth session logged from {IPADDR}."],
    "SSN": ["SSN {SSN} verified for billing.", "Social security number: {SSN}."],
    "MEDICALRECORD": ["Medical record number {MEDICALRECORD}.", "MRN {MEDICALRECORD} confirmed at check-in."],
    "HEALTHPLAN": ["Health plan beneficiary number {HEALTHPLAN}.", "Member ID {HEALTHPLAN}."],
    "ACCOUNT": ["Account number {ACCOUNT} on file.", "Billing account {ACCOUNT}."],
    "LICENSE": ["Driver's license {LICENSE} presented as identification.", "License number {LICENSE}."],
    "VEHICLE": ["Vehicle plate {VEHICLE} noted by EMS.", "Arrived in a car with plate {VEHICLE}."],
    "DEVICE": ["Pacemaker serial number {DEVICE}.", "Device ID {DEVICE} interrogated today."],
    "BIOID": ["Biometric identifier {BIOID} enrolled.", "Fingerprint record {BIOID}."],
    "IDNUM": ["Study ID {IDNUM}.", "Encounter number {IDNUM}."],
    "INSTITUTIONAL": ["Seen in {INSTITUTIONAL}.", "Transferred to {INSTITUTIONAL} overnight."],
}

Filler = Callable[[str, str | None], tuple[str, str]]


def render(template: str, fill: Filler) -> tuple[str, list[Span]]:
    """Fill placeholders with ``fill(label, form) -> (surface, labeled substring)``; return text and spans."""
    out: list[str] = []
    spans: list[Span] = []
    pos = 0
    length = 0
    for m in PLACEHOLDER.finditer(template):
        chunk = template[pos : m.start()]
        out.append(chunk)
        length += len(chunk)
        label, form = m.group(1), m.group(2)
        surface, labeled = fill(label, form)
        if labeled not in surface:
            raise ValueError(f"labeled substring {labeled!r} not inside surface {surface!r}")
        off = surface.index(labeled)
        spans.append(Span(length + off, length + off + len(labeled), label, labeled, "gold"))
        out.append(surface)
        length += len(surface)
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), spans
