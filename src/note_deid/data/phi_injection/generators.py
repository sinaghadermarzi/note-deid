"""Per-subtype synthetic PHI generators with document-level consistency (one patient bundle per note)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from note_deid.labels import I2B2_2014_SUBTYPES

HOSPITALS = [
    "Mercy General Hospital",
    "St. Luke's Medical Center",
    "Riverside Community Hospital",
    "Northgate Regional",
    "Cedar Valley Clinic",
    "Bayview Memorial Hospital",
    "Lakeshore University Hospital",
    "Pinecrest Medical Center",
    "Harborview Medical Center",
    "Summit Ridge Hospital",
    "Oakwood Children's Hospital",
    "Eastbrook Family Practice",
]
ORGANIZATIONS = [
    "Northwind Logistics",
    "Blue Harbor Insurance",
    "Greenfield Public Schools",
    "Apex Manufacturing",
    "Sunrise Senior Living",
    "Metro Transit Authority",
    "Kestrel Software",
    "Highland Dairy Cooperative",
]
PROFESSIONS = [
    "welder",
    "schoolteacher",
    "truck driver",
    "software engineer",
    "registered nurse",
    "electrician",
    "accountant",
    "farmer",
    "retired postal worker",
    "graduate student",
    "chef",
    "firefighter",
    "pharmacist",
    "carpenter",
    "librarian",
    "flight attendant",
]
LOCATION_OTHER = ["Riverside Mall", "Lincoln Park", "the Harbor Bridge", "Cedar Lake", "Union Station"]
INSTITUTIONAL = ["Clinic 4B", "Bldg 12", "Unit 7 West", "MOB-3", "Tower C Suite 210", "Ward B2", "Rm 4415"]
COUNTRIES = ["Canada", "Mexico", "Brazil", "Germany", "Nigeria", "India", "Vietnam", "Ireland", "Peru"]
US_STATES = ["MA", "CA", "TX", "NY", "IL", "WA", "FL", "OH", "PA", "MI", "Massachusetts", "California", "Texas"]

DATE_FORMATS = ["%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%-m/%-d/%Y", "%d %B %Y"]
PARTIAL_DATE_FORMATS = ["%B %Y", "%b %d", "%B %d", "%Y"]


@dataclass
class Person:
    first: str
    last: str
    initial: str = ""

    @property
    def full(self) -> str:
        return f"{self.first} {self.last}"

    def variants(self, role: str) -> list[tuple[str, str]]:
        """(surface text, labeled substring) forms; titles stay outside the labeled substring (i2b2 convention)."""
        if role == "DOCTOR":
            return [
                (f"Dr. {self.last}", self.last),
                (f"Dr. {self.full}", self.full),
                (f"{self.first[0]}. {self.last}, MD", f"{self.first[0]}. {self.last}"),
                (f"{self.full}, MD", self.full),
            ]
        return [
            (self.full, self.full),
            (f"Mr. {self.last}", self.last),
            (self.first, self.first),
            (f"{self.last}, {self.first}", f"{self.last}, {self.first}"),
            (
                f"{self.first} {self.initial}. {self.last}" if self.initial else self.full,
                f"{self.first} {self.initial}. {self.last}" if self.initial else self.full,
            ),
        ]


@dataclass
class PatientBundle:
    patient: Person
    doctors: list[Person]
    hospital: str
    organization: str
    profession: str
    age: int
    dob: date
    visit: date
    discharge: date
    mrn: str
    phone: str
    fax: str
    email: str
    url: str
    street: str
    city: str
    state: str
    zip: str
    country: str
    ssn: str
    healthplan: str
    account: str
    license: str
    vehicle: str
    device: str
    bioid: str
    idnum: str
    username: str
    ipaddr: str
    location_other: str
    institutional: str
    extra: dict[str, str] = field(default_factory=dict)

    def strings(self) -> dict[str, list[str]]:
        """Labeled strings that may be inserted, per subtype (used to check collisions with the source text)."""
        return {
            "PATIENT": [self.patient.first, self.patient.last],
            "DOCTOR": [d.last for d in self.doctors],
            "HOSPITAL": [self.hospital],
            "ORGANIZATION": [self.organization],
            "PROFESSION": [self.profession],
        }


class Generators:
    def __init__(self, seed: int = 0, locales: tuple[str, ...] = ("en_US",)) -> None:
        from faker import Faker

        self.rng = random.Random(seed)
        self.fake = Faker(list(locales))
        self.fake.seed_instance(seed)

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _person(self) -> Person:
        first = self.fake.first_name()
        last = self.fake.last_name()
        initial = self.rng.choice("ABCDEFGHJKLMNPRSTW") if self.rng.random() < 0.3 else ""
        return Person(first, last, initial)

    def _digits(self, n: int) -> str:
        return "".join(self.rng.choice("0123456789") for _ in range(n))

    def _alnum(self, n: int) -> str:
        return "".join(self.rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(n))

    def bundle(self) -> PatientBundle:
        patient = self._person()
        visit = date(2010, 1, 1) + timedelta(days=self.rng.randrange(0, 365 * 15))
        age = self.rng.choice(
            [self.rng.randrange(1, 90), self.rng.randrange(90, 104)]
            if self.rng.random() < 0.08
            else [self.rng.randrange(18, 90)]
        )
        dob = visit - timedelta(days=age * 365 + self.rng.randrange(0, 364))
        email_domain = self.fake.free_email_domain()
        tld = self.rng.choice(["org", "com", "net"])
        return PatientBundle(
            patient=patient,
            doctors=[self._person() for _ in range(self.rng.randrange(1, 4))],
            hospital=self.rng.choice(HOSPITALS),
            organization=self.rng.choice(ORGANIZATIONS),
            profession=self.rng.choice(PROFESSIONS),
            age=age,
            dob=dob,
            visit=visit,
            discharge=visit + timedelta(days=self.rng.randrange(1, 12)),
            mrn=self._digits(self.rng.choice([6, 7, 8])),
            phone=self.fake.numerify(self.rng.choice(["(###) ###-####", "###-###-####", "###.###.####"])),
            fax=self.fake.numerify(self.rng.choice(["(###) ###-####", "###-###-####"])),
            email=f"{patient.first.lower()}.{patient.last.lower()}{self.rng.randrange(1, 99)}@{email_domain}",
            url=f"https://www.{patient.last.lower()}{self.rng.choice(['family', 'health', 'care'])}.{tld}",
            street=self.fake.street_address(),
            city=self.fake.city(),
            state=self.rng.choice(US_STATES),
            zip=self.fake.postcode(),
            country=self.rng.choice(COUNTRIES),
            ssn=self.fake.ssn(),
            healthplan=self._alnum(3) + self._digits(9),
            account=self._digits(self.rng.choice([8, 10, 12])),
            license=self.rng.choice(["D", "L", "S"]) + self._digits(8),
            vehicle=self.fake.license_plate(),
            device=self.rng.choice(["SN", "DEV-", "IMPL"]) + self._alnum(8),
            bioid=self.rng.choice(["FP-", "RET-", "BIO"]) + self._alnum(10),
            idnum=self._alnum(2) + "-" + self._digits(6),
            username=f"{patient.first[0].lower()}{patient.last.lower()}{self._digits(2)}",
            ipaddr=self.fake.ipv4(),
            location_other=self.rng.choice(LOCATION_OTHER),
            institutional=self.rng.choice(INSTITUTIONAL),
        )

    # -- mention rendering -----------------------------------------------------------------------------------------
    def date_text(self, d: date, partial: bool = False) -> str:
        fmt = self.rng.choice(PARTIAL_DATE_FORMATS if partial else DATE_FORMATS)
        try:
            return d.strftime(fmt)
        except ValueError:  # platform without %-m support
            return d.strftime("%m/%d/%Y")

    def mention(self, label: str, b: PatientBundle, form: str | None = None) -> tuple[str, str]:
        """(surface text, labeled substring) for a subtype; ``form`` selects a variant where meaningful."""
        r = self.rng
        if label == "PATIENT":
            v = b.patient.variants("PATIENT")
            return v[0] if form == "full" else (v[2] if form == "first" else r.choice(v))
        if label == "DOCTOR":
            v = r.choice(b.doctors).variants("DOCTOR")
            return v[1] if form == "full" else r.choice(v)
        if label == "USERNAME":
            return b.username, b.username
        if label == "PROFESSION":
            return b.profession, b.profession
        if label == "HOSPITAL":
            return b.hospital, b.hospital
        if label == "ORGANIZATION":
            return b.organization, b.organization
        if label == "STREET":
            return b.street, b.street
        if label == "CITY":
            return b.city, b.city
        if label == "STATE":
            return b.state, b.state
        if label == "COUNTRY":
            return b.country, b.country
        if label == "ZIP":
            return b.zip, b.zip
        if label == "LOCATION-OTHER":
            return b.location_other, b.location_other
        if label == "AGE":
            return str(b.age), str(b.age)
        if label == "DATE":
            which = form or r.choice(["visit", "visit", "dob", "discharge", "followup", "partial"])
            if which == "dob":
                d = b.dob
            elif which == "discharge":
                d = b.discharge
            elif which == "followup":
                d = b.discharge + timedelta(days=7 * r.randrange(1, 9))
            elif which == "partial":
                t = self.date_text(b.visit, partial=True)
                return t, t
            else:
                d = b.visit
            t = self.date_text(d)
            return t, t
        if label == "PHONE":
            return b.phone, b.phone
        if label == "FAX":
            return b.fax, b.fax
        if label == "EMAIL":
            return b.email, b.email
        if label == "URL":
            return b.url, b.url
        if label == "IPADDR":
            return b.ipaddr, b.ipaddr
        if label == "SSN":
            return b.ssn, b.ssn
        if label == "MEDICALRECORD":
            return b.mrn, b.mrn
        if label == "HEALTHPLAN":
            return b.healthplan, b.healthplan
        if label == "ACCOUNT":
            return b.account, b.account
        if label == "LICENSE":
            return b.license, b.license
        if label == "VEHICLE":
            return b.vehicle, b.vehicle
        if label == "DEVICE":
            return b.device, b.device
        if label == "BIOID":
            return b.bioid, b.bioid
        if label == "IDNUM":
            return b.idnum, b.idnum
        if label == "INSTITUTIONAL":
            return b.institutional, b.institutional
        raise KeyError(label)


SUPPORTED_LABELS: tuple[str, ...] = (*I2B2_2014_SUBTYPES, "INSTITUTIONAL")
