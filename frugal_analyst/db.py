"""Synthetic SQLite databases, one per domain.

Everything here is deterministic: the data comes from `random.Random(seed)`
and never from the clock. The same seed always gives the same rows.

Each domain has "coded" columns. The stored value is not the plain English
meaning (for example `CXL_USR` for "cancelled by the customer"). A lookup
table or a `SELECT DISTINCT` reveals the codes. Level 3 tasks depend on this.
"""

from __future__ import annotations

import os
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DOMAINS = ["retail", "clinic", "logistics", "school", "saas"]
TRAIN_DOMAINS = DOMAINS[:4]
OOD_DOMAINS = DOMAINS[4:]  # held out: only used for the OOD test split

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_DIR = REPO_ROOT / "data" / "dbs"

# (table, column) pairs that hold codes. Used by tests and by humans.
CODED_COLUMNS = {
    "retail": [("orders", "status"), ("orders", "channel")],
    "clinic": [("appointments", "status_code"), ("doctors", "specialty_code"), ("patients", "insurance_code")],
    "logistics": [("shipments", "status_code"), ("shipments", "priority_code"), ("carriers", "mode_code")],
    "school": [("enrollments", "result_code"), ("courses", "dept_code"), ("students", "scholarship_code")],
    "saas": [("tickets", "status_code"), ("tickets", "severity_code"), ("invoices", "state_code"), ("accounts", "plan_code")],
}

# ---------------------------------------------------------------------------
# Shared value pools. tasks.py imports these so questions match the data.
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Aarav", "Maya", "Liam", "Sofia", "Noah", "Priya", "Ethan", "Chen", "Olivia", "Mateo",
    "Amara", "Lucas", "Zara", "Hiro", "Isla", "Omar", "Elena", "Kofi", "Nora", "Rafael",
    "Leila", "Jonas", "Mei", "Tariq", "Freya", "Diego", "Anika", "Samuel", "Yuki", "Hannah",
    "Ivan", "Nadia", "Caleb", "Ines", "Arjun", "Clara", "Emeka", "Thea", "Pablo", "Rina",
]
LAST_NAMES = [
    "Patel", "Nguyen", "Garcia", "Okafor", "Schmidt", "Tanaka", "Rossi", "Haddad", "Kowalski", "Silva",
    "Kim", "Johansson", "Moreau", "Singh", "Alvarez", "Bauer", "Costa", "Dubois", "Eriksen", "Farouk",
    "Gupta", "Hughes", "Ivanov", "Jensen", "Khan", "Lopez", "Meyer", "Novak", "Ortiz", "Petrov",
    "Quinn", "Reyes", "Sato", "Torres", "Umar", "Vargas", "Wagner", "Xu", "Yilmaz", "Zhou",
]
CITIES = [
    "Austin", "Boston", "Chicago", "Denver", "Houston", "Seattle", "Portland", "Phoenix", "Atlanta", "Miami",
    "Toronto", "Montreal", "Vancouver", "London", "Manchester", "Dublin", "Berlin", "Munich", "Paris", "Lyon",
    "Madrid", "Lisbon", "Milan", "Rome", "Vienna", "Zurich", "Oslo", "Stockholm", "Mumbai", "Singapore",
]
COUNTRIES = [
    "United States", "Canada", "United Kingdom", "Germany", "France", "Spain", "Italy", "Netherlands",
    "Sweden", "India", "Japan", "Brazil", "Australia", "Mexico", "Singapore",
]

# retail
RETAIL_CATEGORIES = {
    "Electronics": (["Headphones", "Speaker", "Webcam", "Charger", "Keyboard", "Monitor", "Router", "Tablet"], (25, 400)),
    "Kitchen": (["Blender", "Kettle", "Toaster", "Pan", "Knife Set", "Mixer", "Grill", "Scale"], (10, 150)),
    "Garden": (["Hose", "Shovel", "Planter", "Sprinkler", "Pruner", "Lawn Mower", "Rake", "Lantern"], (8, 300)),
    "Sports": (["Yoga Mat", "Dumbbell", "Tent", "Backpack", "Helmet", "Football", "Racket", "Bottle"], (9, 220)),
    "Toys": (["Puzzle", "Robot", "Train Set", "Doll", "Kite", "Board Game", "Drone", "Blocks"], (6, 120)),
    "Books": (["Atlas", "Cookbook", "Novel", "Journal", "Biography", "Guidebook", "Almanac", "Anthology"], (5, 60)),
}
RETAIL_BRANDS = [
    "Aero", "Brio", "Cobalt", "Delta", "Ember", "Fjord", "Gala", "Halo", "Iris", "Juno",
    "Kite", "Lumen", "Maple", "Nova", "Orbit", "Pixel", "Quartz", "Rivet", "Sable", "Tundra",
]
RETAIL_SEGMENTS = ["Consumer", "Small Business", "Corporate"]
RETAIL_STATUS = {  # code -> (description, weight)
    "DLV_OK": ("Delivered", 70),
    "SHP_TRN": ("Shipped, in transit", 5),
    "CXL_USR": ("Cancelled by customer", 9),
    "CXL_OOS": ("Cancelled, item out of stock", 5),
    "RTN_REQ": ("Return requested", 6),
    "PAY_FAIL": ("Payment failed", 5),
}
RETAIL_CHANNELS = {"ONL_WEB": 40, "ONL_APP": 30, "STORE_POS": 22, "PHONE_ORD": 8}

# clinic
CLINIC_SPECIALTIES = {"CARDIO": (180, 12), "DERMA": (120, 14), "ORTHO": (160, 14), "PAEDS": (90, 20), "NEURO": (200, 8), "GEN_PRAC": (70, 32)}
CLINIC_INSURANCE = {"PRIV_INS": 38, "GOV_SCHEME": 30, "SELF_PAY": 17, "EMP_PLAN": 15}
CLINIC_STATUS = {  # code -> (meaning, weight)
    "ATT": ("Attended", 60),
    "NSH": ("No-show: patient did not arrive", 12),
    "CXP": ("Cancelled by the patient", 12),
    "CXD": ("Cancelled by the doctor", 5),
    "RSC": ("Rescheduled to another date", 8),
    "WLT": ("Placed on the waiting list", 3),
}

# logistics
LOGISTICS_REGIONS = ["North", "South", "East", "West", "Central"]
LOGISTICS_MODES = {"AIR_X": (95, 10), "SEA_F": (30, 25), "ROAD_T": (55, 40), "RAIL_F": (40, 25)}  # base cost, weight
LOGISTICS_PRIORITY = {"PRI_EXP": 15, "PRI_STD": 60, "PRI_ECO": 25}
LOGISTICS_STATUS = {
    "DLV": ("Delivered on time", 66),
    "TRN": ("In transit", 8),
    "DLY_WX": ("Delayed by weather", 9),
    "DLY_CU": ("Held at customs", 8),
    "LST": ("Lost in transit", 3),
    "RTS": ("Returned to sender", 6),
}
CARRIER_SUFFIXES = ["Freight", "Logistics", "Haulage", "Express", "Cargo", "Lines", "Transport", "Movers"]

# school
SCHOOL_DEPTS = {"DEP_MTH": 22, "DEP_SCI": 22, "DEP_HUM": 22, "DEP_ART": 12, "DEP_PE": 10, "DEP_TEC": 12}
SCHOOL_SCHOLARSHIPS = {"SCH_NONE": 62, "SCH_MERIT": 16, "SCH_NEED": 14, "SCH_SPORT": 8}
SCHOOL_RESULTS = {  # code -> (meaning, weight)
    "P": ("Passed", 58),
    "D": ("Passed with distinction", 12),
    "F": ("Failed", 14),
    "W": ("Withdrew from the course", 10),
    "I": ("Incomplete, work outstanding", 6),
}
SCHOOL_TERMS = ["2023-Spring", "2023-Fall", "2024-Spring", "2024-Fall"]
COURSE_PREFIXES = [
    "Foundations of", "Introduction to", "Applied", "Advanced", "Topics in", "Principles of", "Workshop in",
    "Seminar on", "Practical", "Studies in", "Essentials of", "Honors", "Modern", "Exploring", "Methods of",
]
COURSE_SUBJECTS = [
    "Algebra", "Geometry", "Statistics", "Physics", "Chemistry", "Biology", "Literature", "History",
    "Drawing", "Music", "Fitness", "Robotics", "Coding", "Economics", "Debate", "Geography",
    "Writing", "Ecology", "Philosophy", "Astronomy",
]

# saas
SAAS_PLANS = {"PLN_FR": ("Free", 40, 0), "PLN_PRO": ("Pro", 30, 49), "PLN_BIZ": ("Business", 20, 199), "PLN_ENT": ("Enterprise", 10, 999)}
SAAS_TICKET_STATUS = {
    "OPN": ("Open", 10),
    "ESC_ENG": ("Escalated to engineering", 10),
    "RSV_FIX": ("Resolved with a fix", 48),
    "RSV_WNF": ("Closed as won't fix", 12),
    "RSV_DUP": ("Closed as duplicate", 20),
}
SAAS_SEVERITY = {
    "S1": ("Critical outage", 6),
    "S2": ("Major degradation", 18),
    "S3": ("Minor issue", 46),
    "S4": ("Cosmetic or how-to question", 30),
}
SAAS_INVOICE_STATE = {"PD": ("Paid", 80), "OVD": ("Overdue", 10), "VOID": ("Voided", 4), "RFD": ("Refunded", 6)}
SAAS_ADJECTIVES = [
    "Bright", "Blue", "Swift", "Quiet", "Golden", "Iron", "Silver", "Rapid", "Clear", "North",
    "Prime", "Solid", "Urban", "Vivid", "Lunar", "Open", "Stellar", "Keen", "Fresh", "Grand",
    "Atlas", "Cedar", "Coral", "Dawn", "Echo", "Forge", "Glass", "Harbor", "Ivory", "Jade",
]
SAAS_NOUNS = [
    "Labs", "Works", "Systems", "Cloud", "Analytics", "Software", "Robotics", "Metrics", "Studio", "Networks",
    "Data", "Logic", "Signal", "Stack", "Grid", "Loop", "Bridge", "Pulse", "Mind", "Forge",
    "Flow", "Scale", "Nest", "Beam", "Shift", "Link", "Wave", "Core", "Lift", "Spark",
]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _names(rng: random.Random, n: int) -> list[str]:
    """n distinct person names (so a text answer is never ambiguous)."""
    picks = rng.sample(range(len(FIRST_NAMES) * len(LAST_NAMES)), n)
    return [f"{FIRST_NAMES[p // len(LAST_NAMES)]} {LAST_NAMES[p % len(LAST_NAMES)]}" for p in picks]


def _dates(rng: random.Random, start: date, days: int, n: int) -> list[str]:
    """n ISO dates, sorted, so that ids grow with time like real event logs."""
    return [d.isoformat() for d in sorted(start + timedelta(days=rng.randrange(days)) for _ in range(n))]


def _one_date(rng: random.Random, start: date, days: int) -> str:
    return (start + timedelta(days=rng.randrange(days))).isoformat()


def _weighted(rng: random.Random, weights: dict[str, int], k: int) -> list[str]:
    return rng.choices(list(weights), weights=list(weights.values()), k=k)


def _insert(conn: sqlite3.Connection, table: str, rows: list[tuple]) -> None:
    marks = ",".join("?" * len(rows[0]))
    conn.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)


# 2023-01-01 to 2024-12-31 is 731 days
_EVENT_START = date(2023, 1, 1)
_EVENT_DAYS = 731

# ---------------------------------------------------------------------------
# One builder per domain
# ---------------------------------------------------------------------------


def _build_retail(conn: sqlite3.Connection, rng: random.Random) -> None:
    conn.executescript(
        """
        CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, segment TEXT, signup_date TEXT);
        CREATE TABLE products(product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, unit_price REAL);
        CREATE TABLE orders(order_id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(customer_id),
            product_id INTEGER REFERENCES products(product_id),
            order_date TEXT, quantity INTEGER, discount_pct INTEGER, status TEXT, channel TEXT);
        CREATE TABLE status_codes(code TEXT PRIMARY KEY, description TEXT);
        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_product ON orders(product_id);
        CREATE INDEX idx_orders_status ON orders(status);
        """
    )
    n_cust, n_prod, n_orders = 800, 300, 3000
    names = _names(rng, n_cust)
    customers = [
        (i + 1, names[i], rng.choice(CITIES), rng.choice(RETAIL_SEGMENTS), _one_date(rng, date(2022, 1, 1), 1095))
        for i in range(n_cust)
    ]
    combos = [(cat, brand, item) for cat, (items, _) in RETAIL_CATEGORIES.items() for brand in RETAIL_BRANDS for item in items]
    products = []
    for i, (cat, brand, item) in enumerate(rng.sample(combos, n_prod)):
        lo, hi = RETAIL_CATEGORIES[cat][1]
        products.append((i + 1, f"{brand} {item}", cat, round(rng.uniform(lo, hi), 2)))
    weights = [1 + rng.random() * 4 for _ in range(n_cust)]  # some customers order more
    buyers = rng.choices(range(1, n_cust + 1), weights=weights, k=n_orders)
    dates = _dates(rng, _EVENT_START, _EVENT_DAYS, n_orders)
    statuses = _weighted(rng, {c: w for c, (_, w) in RETAIL_STATUS.items()}, n_orders)
    channels = _weighted(rng, RETAIL_CHANNELS, n_orders)
    orders = [
        (
            i + 1, buyers[i], rng.randint(1, n_prod), dates[i],
            rng.choices([1, 2, 3, 4, 5], [40, 25, 15, 12, 8])[0],
            rng.choices([0, 5, 10, 15, 20], [50, 20, 15, 10, 5])[0],
            statuses[i], channels[i],
        )
        for i in range(n_orders)
    ]
    _insert(conn, "customers", customers)
    _insert(conn, "products", products)
    _insert(conn, "orders", orders)
    _insert(conn, "status_codes", [(c, d) for c, (d, _) in RETAIL_STATUS.items()])


def _build_clinic(conn: sqlite3.Connection, rng: random.Random) -> None:
    conn.executescript(
        """
        CREATE TABLE patients(patient_id INTEGER PRIMARY KEY, name TEXT, birth_date TEXT, sex TEXT, city TEXT, insurance_code TEXT);
        CREATE TABLE doctors(doctor_id INTEGER PRIMARY KEY, name TEXT, specialty_code TEXT, clinic_city TEXT);
        CREATE TABLE appointments(appt_id INTEGER PRIMARY KEY,
            patient_id INTEGER REFERENCES patients(patient_id),
            doctor_id INTEGER REFERENCES doctors(doctor_id),
            appt_date TEXT, visit_type TEXT, fee REAL, status_code TEXT);
        CREATE TABLE appt_status(code TEXT PRIMARY KEY, meaning TEXT);
        CREATE INDEX idx_appt_patient ON appointments(patient_id);
        CREATE INDEX idx_appt_doctor ON appointments(doctor_id);
        CREATE INDEX idx_appt_status ON appointments(status_code);
        """
    )
    n_pat, n_doc, n_appt = 900, 300, 3000
    pnames = _names(rng, n_pat)
    insurance = _weighted(rng, CLINIC_INSURANCE, n_pat)
    patients = [
        (i + 1, pnames[i], _one_date(rng, date(1940, 1, 1), 29000), rng.choice(["F", "M"]), rng.choice(CITIES), insurance[i])
        for i in range(n_pat)
    ]
    dnames = _names(rng, n_doc)
    specs = rng.choices(list(CLINIC_SPECIALTIES), weights=[w for _, w in CLINIC_SPECIALTIES.values()], k=n_doc)
    doctors = [(i + 1, dnames[i], specs[i], rng.choice(CITIES)) for i in range(n_doc)]
    spec_of = {d[0]: d[2] for d in doctors}
    dates = _dates(rng, _EVENT_START, _EVENT_DAYS, n_appt)
    statuses = _weighted(rng, {c: w for c, (_, w) in CLINIC_STATUS.items()}, n_appt)
    appts = []
    for i in range(n_appt):
        doc = rng.randint(1, n_doc)
        visit = rng.choices(["IN_PERSON", "TELE", "HOME_VISIT"], [70, 22, 8])[0]
        fee = CLINIC_SPECIALTIES[spec_of[doc]][0] * rng.uniform(0.7, 1.4) * (0.8 if visit == "TELE" else 1.0)
        appts.append((i + 1, rng.randint(1, n_pat), doc, dates[i], visit, round(fee, 2), statuses[i]))
    _insert(conn, "patients", patients)
    _insert(conn, "doctors", doctors)
    _insert(conn, "appointments", appts)
    _insert(conn, "appt_status", [(c, m) for c, (m, _) in CLINIC_STATUS.items()])


def _build_logistics(conn: sqlite3.Connection, rng: random.Random) -> None:
    conn.executescript(
        """
        CREATE TABLE depots(depot_id INTEGER PRIMARY KEY, name TEXT, city TEXT, region TEXT);
        CREATE TABLE carriers(carrier_id INTEGER PRIMARY KEY, name TEXT, mode_code TEXT);
        CREATE TABLE shipments(shipment_id INTEGER PRIMARY KEY,
            depot_id INTEGER REFERENCES depots(depot_id),
            carrier_id INTEGER REFERENCES carriers(carrier_id),
            ship_date TEXT, weight_kg REAL, cost REAL, priority_code TEXT, status_code TEXT);
        CREATE TABLE status_codes(code TEXT PRIMARY KEY, meaning TEXT);
        CREATE INDEX idx_ship_depot ON shipments(depot_id);
        CREATE INDEX idx_ship_carrier ON shipments(carrier_id);
        CREATE INDEX idx_ship_status ON shipments(status_code);
        """
    )
    n_depot, n_car, n_ship = 300, 300, 3000
    depots = []
    for i in range(n_depot):
        city = rng.choice(CITIES)
        depots.append((i + 1, f"{city} Hub {i + 1}", city, LOGISTICS_REGIONS[CITIES.index(city) % 5]))
    combos = [f"{last} {suffix}" for last in LAST_NAMES for suffix in CARRIER_SUFFIXES]
    modes = rng.choices(list(LOGISTICS_MODES), weights=[w for _, w in LOGISTICS_MODES.values()], k=n_car)
    carriers = [(i + 1, name, modes[i]) for i, name in enumerate(rng.sample(combos, n_car))]
    mode_of = {c[0]: c[2] for c in carriers}
    dates = _dates(rng, _EVENT_START, _EVENT_DAYS, n_ship)
    prios = _weighted(rng, LOGISTICS_PRIORITY, n_ship)
    statuses = _weighted(rng, {c: w for c, (_, w) in LOGISTICS_STATUS.items()}, n_ship)
    ships = []
    for i in range(n_ship):
        car = rng.randint(1, n_car)
        weight = round(min(400.0, rng.paretovariate(1.5) * 4), 1)
        base = LOGISTICS_MODES[mode_of[car]][0]
        cost = (base + weight * rng.uniform(0.4, 1.2)) * (1.5 if prios[i] == "PRI_EXP" else 1.0)
        ships.append((i + 1, rng.randint(1, n_depot), car, dates[i], weight, round(cost, 2), prios[i], statuses[i]))
    _insert(conn, "depots", depots)
    _insert(conn, "carriers", carriers)
    _insert(conn, "shipments", ships)
    _insert(conn, "status_codes", [(c, m) for c, (m, _) in LOGISTICS_STATUS.items()])


def _build_school(conn: sqlite3.Connection, rng: random.Random) -> None:
    conn.executescript(
        """
        CREATE TABLE students(student_id INTEGER PRIMARY KEY, name TEXT, grade_level INTEGER, enrol_date TEXT, scholarship_code TEXT);
        CREATE TABLE courses(course_id INTEGER PRIMARY KEY, title TEXT, dept_code TEXT, credits INTEGER);
        CREATE TABLE enrollments(enrollment_id INTEGER PRIMARY KEY,
            student_id INTEGER REFERENCES students(student_id),
            course_id INTEGER REFERENCES courses(course_id),
            term TEXT, score REAL, result_code TEXT);
        CREATE TABLE result_codes(code TEXT PRIMARY KEY, meaning TEXT);
        CREATE INDEX idx_enr_student ON enrollments(student_id);
        CREATE INDEX idx_enr_course ON enrollments(course_id);
        CREATE INDEX idx_enr_result ON enrollments(result_code);
        """
    )
    n_stu, n_course, n_enr = 800, 300, 3000
    snames = _names(rng, n_stu)
    schol = _weighted(rng, SCHOOL_SCHOLARSHIPS, n_stu)
    students = [(i + 1, snames[i], rng.randint(9, 12), _one_date(rng, date(2020, 1, 1), 1827), schol[i]) for i in range(n_stu)]
    titles = [f"{p} {s}" for p in COURSE_PREFIXES for s in COURSE_SUBJECTS]
    depts = _weighted(rng, SCHOOL_DEPTS, n_course)
    courses = [(i + 1, t, depts[i], rng.choices([1, 2, 3, 4], [30, 35, 25, 10])[0]) for i, t in enumerate(rng.sample(titles, n_course))]
    terms = sorted(rng.choices(SCHOOL_TERMS, k=n_enr), key=SCHOOL_TERMS.index)
    results = _weighted(rng, {c: w for c, (_, w) in SCHOOL_RESULTS.items()}, n_enr)
    score_range = {"P": (55, 90), "D": (90, 100), "F": (20, 55)}
    enrollments = []
    for i in range(n_enr):
        res = results[i]
        score = round(rng.uniform(*score_range[res]), 2) if res in score_range else None  # W and I have no score
        enrollments.append((i + 1, rng.randint(1, n_stu), rng.randint(1, n_course), terms[i], score, res))
    _insert(conn, "students", students)
    _insert(conn, "courses", courses)
    _insert(conn, "enrollments", enrollments)
    _insert(conn, "result_codes", [(c, m) for c, (m, _) in SCHOOL_RESULTS.items()])


def _build_saas(conn: sqlite3.Connection, rng: random.Random) -> None:
    # This domain is held out for the OOD test. It also uses one generic
    # code_book table instead of one lookup table per column.
    conn.executescript(
        """
        CREATE TABLE accounts(account_id INTEGER PRIMARY KEY, company TEXT, country TEXT, plan_code TEXT, signup_date TEXT);
        CREATE TABLE tickets(ticket_id INTEGER PRIMARY KEY,
            account_id INTEGER REFERENCES accounts(account_id),
            opened_date TEXT, severity_code TEXT, status_code TEXT, hours_to_resolve REAL);
        CREATE TABLE invoices(invoice_id INTEGER PRIMARY KEY,
            account_id INTEGER REFERENCES accounts(account_id),
            invoice_date TEXT, amount REAL, state_code TEXT);
        CREATE TABLE code_book(kind TEXT, code TEXT, meaning TEXT);
        CREATE INDEX idx_tickets_account ON tickets(account_id);
        CREATE INDEX idx_invoices_account ON invoices(account_id);
        """
    )
    n_acc, n_tick, n_inv = 800, 3000, 3000
    combos = [f"{a} {n}" for a in SAAS_ADJECTIVES for n in SAAS_NOUNS]
    plans = rng.choices(list(SAAS_PLANS), weights=[w for _, w, _ in SAAS_PLANS.values()], k=n_acc)
    accounts = [
        (i + 1, name, rng.choice(COUNTRIES), plans[i], _one_date(rng, date(2022, 1, 1), 1095))
        for i, name in enumerate(rng.sample(combos, n_acc))
    ]
    weights = [1 + rng.random() * 3 for _ in range(n_acc)]
    t_dates = _dates(rng, _EVENT_START, _EVENT_DAYS, n_tick)
    sev = _weighted(rng, {c: w for c, (_, w) in SAAS_SEVERITY.items()}, n_tick)
    stat = _weighted(rng, {c: w for c, (_, w) in SAAS_TICKET_STATUS.items()}, n_tick)
    hours_cap = {"S1": 24, "S2": 72, "S3": 160, "S4": 300}
    tickets = []
    for i in range(n_tick):
        hours = round(rng.uniform(1, hours_cap[sev[i]]), 1) if stat[i].startswith("RSV") else None
        tickets.append((i + 1, rng.choices(range(1, n_acc + 1), weights=weights)[0], t_dates[i], sev[i], stat[i], hours))
    paid_ids = [a[0] for a in accounts if a[3] != "PLN_FR"]
    base_of = {a[0]: SAAS_PLANS[a[3]][2] for a in accounts}
    i_dates = _dates(rng, _EVENT_START, _EVENT_DAYS, n_inv)
    states = _weighted(rng, {c: w for c, (_, w) in SAAS_INVOICE_STATE.items()}, n_inv)
    invoices = []
    for i in range(n_inv):
        acc = rng.choice(paid_ids)
        invoices.append((i + 1, acc, i_dates[i], round(base_of[acc] * rng.uniform(0.8, 3.0), 2), states[i]))
    book = []
    for kind, table in [
        ("plan", {c: p[0] for c, p in SAAS_PLANS.items()}),
        ("ticket_status", {c: v[0] for c, v in SAAS_TICKET_STATUS.items()}),
        ("severity", {c: v[0] for c, v in SAAS_SEVERITY.items()}),
        ("invoice_state", {c: v[0] for c, v in SAAS_INVOICE_STATE.items()}),
    ]:
        book += [(kind, code, meaning) for code, meaning in table.items()]
    _insert(conn, "accounts", accounts)
    _insert(conn, "tickets", tickets)
    _insert(conn, "invoices", invoices)
    _insert(conn, "code_book", book)


_BUILDERS = {
    "retail": _build_retail,
    "clinic": _build_clinic,
    "logistics": _build_logistics,
    "school": _build_school,
    "saas": _build_saas,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_db(domain: str, path: str | Path, seed: int = 0) -> None:
    """Build one domain database at `path`. Same seed gives the same data."""
    if domain not in _BUILDERS:
        raise ValueError(f"unknown domain {domain!r}. Valid: {', '.join(DOMAINS)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build in a temp file and rename, so a reader never sees a half-built file.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        _BUILDERS[domain](conn, random.Random(f"db:{seed}:{domain}"))
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, path)


def db_path(domain: str, db_dir: str | Path | None = None) -> Path:
    """Path of the domain database. FRUGAL_DB_DIR overrides the default dir."""
    if db_dir is None:
        db_dir = os.environ.get("FRUGAL_DB_DIR") or DEFAULT_DB_DIR
    return Path(db_dir) / f"{domain}.sqlite"


def ensure_dbs(db_dir: str | Path | None = None, seed: int = 0, force: bool = False) -> None:
    """Build any missing database. `force=True` rebuilds all of them."""
    for domain in DOMAINS:
        path = db_path(domain, db_dir)
        if force or not path.exists():
            build_db(domain, path, seed)


def ensure_db(domain: str, db_dir: str | Path | None = None) -> Path:
    """Build one database if it is missing and return its path."""
    path = db_path(domain, db_dir)
    if not path.exists():
        build_db(domain, path)
    return path


def schema_text(domain: str, db_dir: str | Path | None = None) -> str:
    """Compact schema: one line per table with column types and the row count."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}. Valid: {', '.join(DOMAINS)}")
    path = ensure_db(domain, db_dir)
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        lines = []
        for table in tables:
            cols = ", ".join(f"{c[1]} {c[2]}" for c in conn.execute(f"PRAGMA table_info({table})"))
            n_rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            lines.append(f"{table}({cols}) -- {n_rows} rows")
        return "\n".join(lines)
    finally:
        conn.close()
