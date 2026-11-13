"""Task generator: templated questions with a gold SQL query for each one.

How it works
------------
A `Template` has several question phrasings, one gold SQL string and a set of
parameters. Each parameter is a list of options. An option has a SQL value
(`value`) and one or more ways to say it in English (`words`). For a coded
value the SQL value is the code and the words are the human meaning.

For every combination of options the generator runs the gold SQL, checks the
answer is usable, and keeps the task. Unusable tasks are dropped: empty or NULL
answers, zero counts, and top-1 questions where two rows tie.

Levels
------
1: one table, filter plus aggregate.
2: join plus aggregate, or group-by with a top-1 answer.
3: the question uses a human meaning, the data stores a code. The efficient
   path is get_schema, one cheap lookup, then the final query.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from . import db
from .db import DOMAINS, OOD_DOMAINS, ensure_dbs
from .env import execute_metered, format_number

SPLITS = ["train", "val", "test_id", "test_ood"]
TRAIN_DOMAIN_TASKS = 140  # per train domain: 100 train, 10 val, 30 test_id
OOD_DOMAIN_TASKS = 100


@dataclass
class Task:
    id: str  # stable, e.g. "retail-L2-0042"
    domain: str
    level: int  # 1, 2 or 3
    template: str
    question: str
    answer_type: str  # "number" or "text"
    gold_sql: str  # final query of the efficient solution
    gold_answer: str  # canonical string
    oracle_calls: list[dict]  # efficient tool-call sequence
    optimal_tool_calls: int
    optimal_rows: int  # total rows returned across oracle run_sql calls
    optimal_vm_steps: int  # total vm steps across oracle run_sql calls
    split: str  # "train" | "val" | "test_id" | "test_ood"


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Opt:
    value: str  # goes into the SQL, and identifies the option
    words: tuple[str, ...]  # ways to say it in the question
    extra: tuple[tuple[str, str], ...] = ()  # more SQL fields, used as {name_key}


def opt(value, words=None, **extra) -> Opt:
    if words is None:
        words = [str(value)]
    elif isinstance(words, str):
        words = [words]
    return Opt(str(value), tuple(words), tuple(extra.items()))


def nums(*values) -> list[Opt]:
    return [opt(v) for v in values]


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
ORDINALS = ["first", "second", "third", "fourth"]


def _month_start(year: int, month: int) -> str:
    year, month = year + (month - 1) // 12, (month - 1) % 12 + 1
    return f"{year}-{month:02d}-01"


def month_opts(years) -> list[Opt]:
    return [
        opt(f"{y}-{m:02d}", [f"in {MONTHS[m - 1]} {y}", f"during {MONTHS[m - 1]} {y}"], start=_month_start(y, m), end=_month_start(y, m + 1))
        for y in years for m in range(1, 13)
    ]


def quarter_opts(years) -> list[Opt]:
    return [
        opt(f"{y}Q{q}", [f"in Q{q} {y}", f"in the {ORDINALS[q - 1]} quarter of {y}"], start=_month_start(y, 3 * q - 2), end=_month_start(y, 3 * q + 1))
        for y in years for q in range(1, 5)
    ]


def half_opts(years) -> list[Opt]:
    return [
        opt(f"{y}H{h}", [f"in the {ORDINALS[h - 1]} half of {y}", f"in H{h} {y}"], start=_month_start(y, 6 * h - 5), end=_month_start(y, 6 * h + 1))
        for y in years for h in (1, 2)
    ]


def year_opts(years) -> list[Opt]:
    return [opt(str(y), [f"in {y}", f"during {y}"], start=f"{y}-01-01", end=f"{y + 1}-01-01") for y in years]


EVENT_YEARS = [2023, 2024]
# Months, quarters and years: 34 periods. COARSE has only quarters and years.
PERIODS = month_opts(EVENT_YEARS) + quarter_opts(EVENT_YEARS) + year_opts(EVENT_YEARS)
COARSE = quarter_opts(EVENT_YEARS) + year_opts(EVENT_YEARS)
YEARS = year_opts(EVENT_YEARS)
SIGNUP_YEARS = year_opts([2022, 2023, 2024])
SCHOOL_PERIODS = year_opts(range(2020, 2025)) + half_opts(range(2020, 2025))

CITY = [opt(c) for c in db.CITIES]
COUNTRY = [opt(c) for c in db.COUNTRIES]
TERMS = [opt(t, [t.replace("-", " "), t]) for t in db.SCHOOL_TERMS]


def when(column: str, prefix: str = "period") -> str:
    """SQL date-range filter whose bounds come from a period option."""
    return f"{column} >= '{{{prefix}_start}}' AND {column} < '{{{prefix}_end}}'"


def top1(key: str, group_by: str, body: str, metric: str) -> tuple[str, str]:
    """Gold SQL for 'which X has the most Y', plus a LIMIT 2 query to detect ties."""
    sql = f"SELECT {key} {body} GROUP BY {group_by} ORDER BY {metric} DESC LIMIT 1"
    tie = f"SELECT {metric} {body} GROUP BY {group_by} ORDER BY {metric} DESC LIMIT 2"
    return sql, tie


@dataclass
class Template:
    domain: str
    name: str
    level: int
    answer_type: str
    questions: list[str]
    sql: str
    params: dict[str, list[Opt]]
    lookup_sql: str | None = None  # level 3 only: the cheap lookup before the final query
    tie_sql: str | None = None  # top-1 only: returns the top two metric values


# ---------------------------------------------------------------------------
# Templates: retail
# ---------------------------------------------------------------------------

REV = "o.quantity * p.unit_price * (1 - o.discount_pct / 100.0)"
R_JOIN_P = "JOIN products p ON p.product_id = o.product_id"
R_JOIN_C = "JOIN customers c ON c.customer_id = o.customer_id"

RETAIL_STATUS_OPTS = [
    opt("DLV_OK", ["delivered", "delivered successfully"]),
    opt("SHP_TRN", ["still in transit", "shipped but not yet delivered"]),
    opt("CXL_USR", ["cancelled by the customer", "cancelled by the buyer"]),
    opt("CXL_OOS", ["cancelled because the item was out of stock", "cancelled for lack of stock"]),
    opt("RTN_REQ", ["flagged with a return request", "marked for return"]),
    opt("PAY_FAIL", ["rejected because the payment failed", "stopped by a failed payment"]),
]
RETAIL_CHANNEL_OPTS = [
    opt("ONL_WEB", ["on the website", "through the web shop"]),
    opt("ONL_APP", ["in the mobile app", "through the shopping app"]),
    opt("STORE_POS", ["in a physical store", "at a shop counter"]),
    opt("PHONE_ORD", ["by phone", "through the call centre"]),
]

_top_city_rev, _top_city_rev_tie = top1(
    "c.city", "c.city",
    f"FROM orders o {R_JOIN_C} {R_JOIN_P} WHERE p.category = '{{category}}' AND {when('o.order_date')}", f"SUM({REV})",
)
_top_cat_city, _top_cat_city_tie = top1(
    "p.category", "p.category",
    f"FROM orders o {R_JOIN_C} {R_JOIN_P} WHERE c.city = '{{city}}' AND {when('o.order_date')}", f"SUM({REV})",
)
_top_cat_status, _top_cat_status_tie = top1(
    "p.category", "p.category",
    f"FROM orders o {R_JOIN_P} WHERE o.status = '{{status}}' AND {when('o.order_date')}", f"SUM({REV})",
)
CATEGORY = [opt(c) for c in db.RETAIL_CATEGORIES]

RETAIL = [
    Template("retail", "orders_qty_period", 1, "number",
        ["How many orders {period} had a quantity of {qty} or more?",
         "Count the orders placed {period} with at least {qty} units.",
         "How many orders {period} included {qty} or more units?"],
        f"SELECT COUNT(*) FROM orders WHERE {when('order_date')} AND quantity >= {{qty}}",
        {"period": PERIODS, "qty": nums(2, 3, 4, 5)}),
    Template("retail", "avg_discount_period", 1, "number",
        ["What was the average discount percentage on orders {period} with {qty} or more units?",
         "Among orders {period} with at least {qty} units, what was the mean discount percentage?"],
        f"SELECT AVG(discount_pct) FROM orders WHERE {when('order_date')} AND quantity >= {{qty}}",
        {"period": PERIODS, "qty": nums(1, 2, 3, 4)}),
    Template("retail", "customers_city_signup", 1, "number",
        ["How many customers from {city} signed up {period}?",
         "How many customers living in {city} joined {period}?",
         "Count the customers in {city} whose signup date falls {period}."],
        f"SELECT COUNT(*) FROM customers WHERE city = '{{city}}' AND {when('signup_date')}",
        {"city": CITY, "period": SIGNUP_YEARS}),
    Template("retail", "revenue_category_period", 2, "number",
        ["What was the total revenue from {category} products {period}, after discounts?",
         "How much revenue did {category} products bring in {period} once discounts are applied?"],
        f"SELECT SUM({REV}) FROM orders o {R_JOIN_P} WHERE p.category = '{{category}}' AND {when('o.order_date')}",
        {"category": CATEGORY, "period": PERIODS}),
    Template("retail", "top_city_category_revenue", 2, "text",
        ["Which city generated the most revenue from {category} products {period}?",
         "Which city's customers spent the most on {category} items {period}?"],
        _top_city_rev, {"category": CATEGORY, "period": COARSE}, tie_sql=_top_city_rev_tie),
    Template("retail", "top_category_city_revenue", 2, "text",
        ["Which product category earned the most revenue from customers in {city} {period}?",
         "Customers in {city} spent the most on which product category {period}?"],
        _top_cat_city, {"city": CITY, "period": COARSE}, tie_sql=_top_cat_city_tie),
    Template("retail", "orders_status_period", 3, "number",
        ["How many orders were {status} {period}?",
         "Count the orders {period} that were {status}.",
         "{period_cap}, how many orders were {status}?"],
        f"SELECT COUNT(*) FROM orders WHERE status = '{{status}}' AND {when('order_date')}",
        {"status": RETAIL_STATUS_OPTS, "period": PERIODS},
        lookup_sql="SELECT code, description FROM status_codes"),
    Template("retail", "orders_channel_city", 3, "number",
        ["How many orders from customers in {city} were placed {channel} {period}?",
         "How many {period} orders did customers in {city} place {channel}?"],
        f"SELECT COUNT(*) FROM orders o {R_JOIN_C} WHERE c.city = '{{city}}' AND o.channel = '{{channel}}' AND {when('o.order_date')}",
        {"city": CITY, "channel": RETAIL_CHANNEL_OPTS, "period": YEARS},
        lookup_sql="SELECT DISTINCT channel FROM orders"),
    Template("retail", "top_category_status_revenue", 3, "text",
        ["Which product category brought in the most revenue from orders that were {status} {period}?",
         "Among orders {status} {period}, which product category had the highest revenue?"],
        _top_cat_status, {"status": RETAIL_STATUS_OPTS, "period": COARSE},
        lookup_sql="SELECT code, description FROM status_codes", tie_sql=_top_cat_status_tie),
]

# ---------------------------------------------------------------------------
# Templates: clinic
# ---------------------------------------------------------------------------

A_JOIN_P = "JOIN patients p ON p.patient_id = a.patient_id"
A_JOIN_D = "JOIN doctors d ON d.doctor_id = a.doctor_id"

CLINIC_STATUS_OPTS = [
    opt("ATT", ["attended by the patient", "kept by the patient"]),
    opt("NSH", ["missed because the patient did not show up", "no-shows"]),
    opt("CXP", ["cancelled by the patient", "called off by the patient"]),
    opt("CXD", ["cancelled by the doctor", "called off by the doctor"]),
    opt("RSC", ["rescheduled", "moved to another date"]),
    opt("WLT", ["placed on the waiting list", "put on the wait list"]),
]
CLINIC_SPECIALTY_OPTS = [
    opt("CARDIO", ["cardiologist", "heart specialist"]),
    opt("DERMA", ["dermatologist", "skin specialist"]),
    opt("ORTHO", ["orthopaedic surgeon", "bone and joint specialist"]),
    opt("PAEDS", ["paediatrician", "children's doctor"]),
    opt("NEURO", ["neurologist", "nerve specialist"]),
    opt("GEN_PRAC", ["general practitioner", "family doctor"]),
]
CLINIC_INSURANCE_OPTS = [
    opt("PRIV_INS", ["private insurance", "a private insurer"]),
    opt("GOV_SCHEME", ["a government health scheme", "public insurance"]),
    opt("SELF_PAY", ["no insurance, paying out of pocket", "self-pay terms"]),
    opt("EMP_PLAN", ["an employer health plan", "insurance through their employer"]),
]

_top_doc_fees, _top_doc_fees_tie = top1(
    "d.name", "d.doctor_id",
    f"FROM appointments a {A_JOIN_D} {A_JOIN_P} WHERE p.city = '{{city}}' AND {when('a.appt_date')}", "SUM(a.fee)",
)
_top_city_fees, _top_city_fees_tie = top1(
    "p.city", "p.city", f"FROM appointments a {A_JOIN_P} WHERE {when('a.appt_date')}", "SUM(a.fee)",
)
_top_spec_status, _top_spec_status_tie = top1(
    "d.name", "d.doctor_id",
    f"FROM appointments a {A_JOIN_D} WHERE a.status_code = '{{status}}' AND {when('a.appt_date')}", "SUM(a.fee)",
)

CLINIC = [
    Template("clinic", "appts_fee_period", 1, "number",
        ["How many appointments {period} had a fee above {fee}?",
         "Count the appointments {period} that charged more than {fee}.",
         "How many appointments {period} cost over {fee}?"],
        f"SELECT COUNT(*) FROM appointments WHERE {when('appt_date')} AND fee > {{fee}}",
        {"period": PERIODS, "fee": nums(60, 80, 100, 120, 150)}),
    Template("clinic", "patients_city_born", 1, "number",
        ["How many patients living in {city} were born in {year} or later?",
         "Count the patients from {city} with a birth year of {year} or later."],
        "SELECT COUNT(*) FROM patients WHERE city = '{city}' AND birth_date >= '{year}-01-01'",
        {"city": CITY, "year": nums(1960, 1970, 1980, 1990, 2000, 2010)}),
    Template("clinic", "total_fees_period", 1, "number",
        ["What was the total of all appointment fees {period}?",
         "How much did appointments charge in total {period}?"],
        f"SELECT SUM(fee) FROM appointments WHERE {when('appt_date')}",
        {"period": PERIODS}),
    Template("clinic", "avg_fee_patient_city", 2, "number",
        ["What was the average fee for appointments {period} by patients living in {city}?",
         "What did appointments {period} cost on average for patients from {city}?"],
        f"SELECT AVG(a.fee) FROM appointments a {A_JOIN_P} WHERE p.city = '{{city}}' AND {when('a.appt_date')}",
        {"city": CITY, "period": COARSE}),
    Template("clinic", "top_doctor_fees_city", 2, "text",
        ["Which doctor collected the most in fees from patients living in {city} {period}?",
         "Which doctor earned the highest total fees from {city} patients {period}?"],
        _top_doc_fees, {"city": CITY, "period": COARSE}, tie_sql=_top_doc_fees_tie),
    Template("clinic", "top_city_fees_period", 2, "text",
        ["Which patient city paid the highest total appointment fees {period}?",
         "Patients from which city were charged the most in fees {period}?"],
        _top_city_fees, {"period": PERIODS}, tie_sql=_top_city_fees_tie),
    Template("clinic", "appts_status_period", 3, "number",
        ["How many appointments were {status} {period}?",
         "Count the appointments {period} that were {status}."],
        f"SELECT COUNT(*) FROM appointments WHERE status_code = '{{status}}' AND {when('appt_date')}",
        {"status": CLINIC_STATUS_OPTS, "period": PERIODS},
        lookup_sql="SELECT code, meaning FROM appt_status"),
    Template("clinic", "appts_specialty_period", 3, "number",
        ["How many appointments {period} were with a {specialty}?",
         "Count the appointments {period} booked with a {specialty}."],
        f"SELECT COUNT(*) FROM appointments a {A_JOIN_D} WHERE d.specialty_code = '{{specialty}}' AND {when('a.appt_date')}",
        {"specialty": CLINIC_SPECIALTY_OPTS, "period": PERIODS},
        lookup_sql="SELECT DISTINCT specialty_code FROM doctors"),
    Template("clinic", "avg_fee_insurance", 3, "number",
        ["What was the average appointment fee {period} for patients with {insurance}?",
         "What did appointments {period} cost on average for patients on {insurance}?"],
        f"SELECT AVG(a.fee) FROM appointments a {A_JOIN_P} WHERE p.insurance_code = '{{insurance}}' AND {when('a.appt_date')}",
        {"insurance": CLINIC_INSURANCE_OPTS, "period": COARSE},
        lookup_sql="SELECT DISTINCT insurance_code FROM patients"),
    Template("clinic", "top_doctor_status_fees", 3, "text",
        ["Which doctor collected the highest total fees from appointments that were {status} {period}?"],
        _top_spec_status, {"status": CLINIC_STATUS_OPTS, "period": COARSE},
        lookup_sql="SELECT code, meaning FROM appt_status", tie_sql=_top_spec_status_tie),
]

# ---------------------------------------------------------------------------
# Templates: logistics
# ---------------------------------------------------------------------------

S_JOIN_D = "JOIN depots d ON d.depot_id = s.depot_id"
S_JOIN_C = "JOIN carriers c ON c.carrier_id = s.carrier_id"

LOGISTICS_STATUS_OPTS = [
    opt("DLV", ["delivered on time", "delivered without delay"]),
    opt("TRN", ["still in transit", "on the road or at sea"]),
    opt("DLY_WX", ["delayed by bad weather", "held up by weather"]),
    opt("DLY_CU", ["held at customs", "stuck in customs"]),
    opt("LST", ["lost in transit", "lost on the way"]),
    opt("RTS", ["returned to the sender", "sent back to the sender"]),
]
LOGISTICS_MODE_OPTS = [
    opt("AIR_X", ["air freight carriers", "carriers that fly cargo"]),
    opt("SEA_F", ["sea freight carriers", "ocean shipping lines"]),
    opt("ROAD_T", ["road trucking carriers", "trucking companies"]),
    opt("RAIL_F", ["rail freight carriers", "railway cargo carriers"]),
]
LOGISTICS_PRIORITY_OPTS = [
    opt("PRI_EXP", ["express", "fast-delivery"]),
    opt("PRI_STD", ["standard", "regular-speed"]),
    opt("PRI_ECO", ["economy", "slow low-cost"]),
]
REGION = [opt(r) for r in db.LOGISTICS_REGIONS]

_top_depot_city, _top_depot_city_tie = top1(
    "d.city", "d.city", f"FROM shipments s {S_JOIN_D} WHERE {when('s.ship_date')}", "SUM(s.weight_kg)",
)
_top_carrier, _top_carrier_tie = top1(
    "c.name", "c.carrier_id",
    f"FROM shipments s {S_JOIN_C} {S_JOIN_D} WHERE d.region = '{{region}}' AND {when('s.ship_date')}", "SUM(s.weight_kg)",
)
_top_region_prio, _top_region_prio_tie = top1(
    "d.region", "d.region",
    f"FROM shipments s {S_JOIN_D} WHERE s.priority_code = '{{priority}}' AND {when('s.ship_date')}", "SUM(s.cost)",
)

LOGISTICS = [
    Template("logistics", "shipments_heavy_period", 1, "number",
        ["How many shipments went out {period} weighing more than {kg} kg?",
         "Count the shipments {period} heavier than {kg} kg."],
        f"SELECT COUNT(*) FROM shipments WHERE {when('ship_date')} AND weight_kg > {{kg}}",
        {"period": PERIODS, "kg": nums(10, 25, 50, 100, 200)}),
    Template("logistics", "avg_cost_period", 1, "number",
        ["What was the average shipping cost {period} for shipments weighing at most {kg} kg?",
         "What was the mean cost of shipments {period} that weighed {kg} kg or less?"],
        f"SELECT AVG(cost) FROM shipments WHERE {when('ship_date')} AND weight_kg <= {{kg}}",
        {"period": PERIODS, "kg": nums(5, 10, 25, 50, 100)}),
    Template("logistics", "depots_city_region", 1, "number",
        ["How many depots are in {city}?",
         "How many depots does the network have in {city}?",
         "Count the depots located in {city}."],
        "SELECT COUNT(*) FROM depots WHERE city = '{city}'",
        {"city": CITY}),
    Template("logistics", "cost_region_period", 2, "number",
        ["What was the total shipping cost of shipments leaving depots in the {region} region {period}?",
         "How much did shipments from {region} region depots cost in total {period}?"],
        f"SELECT SUM(s.cost) FROM shipments s {S_JOIN_D} WHERE d.region = '{{region}}' AND {when('s.ship_date')}",
        {"region": REGION, "period": COARSE}),
    Template("logistics", "top_depot_city_weight", 2, "text",
        ["Which depot city shipped the greatest total weight {period}?",
         "Depots in which city sent out the most weight {period}?"],
        _top_depot_city, {"period": PERIODS}, tie_sql=_top_depot_city_tie),
    Template("logistics", "top_carrier_region_weight", 2, "text",
        ["Which carrier moved the most total weight from depots in the {region} region {period}?",
         "Among shipments leaving {region} region depots {period}, which carrier handled the highest total weight?"],
        _top_carrier, {"region": REGION, "period": COARSE}, tie_sql=_top_carrier_tie),
    Template("logistics", "shipments_status_period", 3, "number",
        ["How many shipments were {status} {period}?",
         "Count the shipments {period} that were {status}."],
        f"SELECT COUNT(*) FROM shipments WHERE status_code = '{{status}}' AND {when('ship_date')}",
        {"status": LOGISTICS_STATUS_OPTS, "period": PERIODS},
        lookup_sql="SELECT code, meaning FROM status_codes"),
    Template("logistics", "cost_mode_period", 3, "number",
        ["What was the total cost of shipments carried by {mode} {period}?",
         "How much did shipments handled by {mode} cost in total {period}?"],
        f"SELECT SUM(s.cost) FROM shipments s {S_JOIN_C} WHERE c.mode_code = '{{mode}}' AND {when('s.ship_date')}",
        {"mode": LOGISTICS_MODE_OPTS, "period": PERIODS},
        lookup_sql="SELECT DISTINCT mode_code FROM carriers"),
    Template("logistics", "top_region_priority_cost", 3, "text",
        ["Which region's depots had the highest total cost of {priority} shipments {period}?",
         "Which depot region spent the most on {priority} shipments {period}?"],
        _top_region_prio, {"priority": LOGISTICS_PRIORITY_OPTS, "period": PERIODS},
        lookup_sql="SELECT DISTINCT priority_code FROM shipments", tie_sql=_top_region_prio_tie),
]

# ---------------------------------------------------------------------------
# Templates: school
# ---------------------------------------------------------------------------

E_JOIN_S = "JOIN students s ON s.student_id = e.student_id"
E_JOIN_C = "JOIN courses c ON c.course_id = e.course_id"

SCHOOL_RESULT_OPTS = [
    opt("P", ["a pass", "a passing result"]),
    opt("D", ["a distinction", "a pass with distinction"]),
    opt("F", ["a fail", "a failing result"]),
    opt("W", ["a withdrawal", "the student leaving the course"]),
    opt("I", ["an incomplete", "unfinished work"]),
]
SCHOOL_DEPT_OPTS = [
    opt("DEP_MTH", ["mathematics", "maths"]),
    opt("DEP_SCI", ["science", "natural science"]),
    opt("DEP_HUM", ["humanities", "arts and letters"]),
    opt("DEP_ART", ["art", "visual arts"]),
    opt("DEP_PE", ["physical education", "sports"]),
    opt("DEP_TEC", ["technology", "computing"]),
]
SCHOOL_SCHOLAR_OPTS = [
    opt("SCH_NONE", ["no scholarship", "no financial aid"]),
    opt("SCH_MERIT", ["a merit scholarship", "an academic merit award"]),
    opt("SCH_NEED", ["a need-based scholarship", "need-based financial aid"]),
    opt("SCH_SPORT", ["a sports scholarship", "an athletic scholarship"]),
]
GRADE = nums(9, 10, 11, 12)
CREDITS = nums(1, 2, 3, 4)

_top_student, _top_student_tie = top1(
    "s.name", "s.student_id",
    f"FROM enrollments e {E_JOIN_S} {E_JOIN_C} WHERE s.grade_level = {{grade}} AND e.term = '{{term}}' AND c.credits = {{credits}} AND e.score IS NOT NULL",
    "AVG(e.score)",
)
_top_course_dept, _top_course_dept_tie = top1(
    "c.title", "c.course_id",
    f"FROM enrollments e {E_JOIN_C} WHERE c.dept_code = '{{dept}}' AND e.term = '{{term}}' AND e.score IS NOT NULL",
    "AVG(e.score)",
)

SCHOOL = [
    Template("school", "students_grade_enrol", 1, "number",
        ["How many students in grade {grade} enrolled {period}?",
         "Count the grade {grade} students whose enrolment date falls {period}."],
        f"SELECT COUNT(*) FROM students WHERE grade_level = {{grade}} AND {when('enrol_date')}",
        {"grade": GRADE, "period": SCHOOL_PERIODS}),
    Template("school", "enrollments_term_score", 1, "number",
        ["How many course enrollments in {term} scored at least {score}?",
         "In {term}, how many enrollments ended with a score of {score} or higher?"],
        "SELECT COUNT(*) FROM enrollments WHERE term = '{term}' AND score >= {score}",
        {"term": TERMS, "score": nums(50, 60, 70, 80, 90, 95)}),
    Template("school", "avg_score_term_floor", 1, "number",
        ["What was the average score in {term} among enrollments that scored at least {score}?",
         "Looking only at {term} enrollments scoring {score} or more, what was the mean score?"],
        "SELECT AVG(score) FROM enrollments WHERE term = '{term}' AND score >= {score}",
        {"term": TERMS, "score": nums(20, 40, 50, 60, 70, 80)}),
    Template("school", "avg_score_grade_credits", 2, "number",
        ["What was the average score of grade {grade} students in {term} in courses worth {credits} credits?",
         "In {term}, what was the mean score of grade {grade} students taking {credits}-credit courses?"],
        f"SELECT AVG(e.score) FROM enrollments e {E_JOIN_S} {E_JOIN_C} WHERE s.grade_level = {{grade}} AND e.term = '{{term}}' AND c.credits = {{credits}}",
        {"grade": GRADE, "term": TERMS, "credits": CREDITS}),
    Template("school", "top_student_grade_credits", 2, "text",
        ["Which grade {grade} student had the highest average score in {term} in courses worth {credits} credits?",
         "Which grade {grade} student scored best on average in {term} among {credits}-credit courses?"],
        _top_student, {"grade": GRADE, "term": TERMS, "credits": CREDITS}, tie_sql=_top_student_tie),
    Template("school", "credits_attempted", 2, "number",
        ["What is the total number of credits attempted by grade {grade} students in {term} with a score of at least {score}?",
         "How many credits in total did grade {grade} students earn scores of {score} or more on in {term}?"],
        f"SELECT SUM(c.credits) FROM enrollments e {E_JOIN_S} {E_JOIN_C} WHERE s.grade_level = {{grade}} AND e.term = '{{term}}' AND e.score >= {{score}}",
        {"grade": GRADE, "term": TERMS, "score": nums(50, 60, 70, 80, 90)}),
    Template("school", "enrollments_result_grade", 3, "number",
        ["How many enrollments by grade {grade} students in {term} ended with {result}?",
         "In {term}, how many grade {grade} students' enrollments finished with {result}?"],
        f"SELECT COUNT(*) FROM enrollments e {E_JOIN_S} WHERE s.grade_level = {{grade}} AND e.term = '{{term}}' AND e.result_code = '{{result}}'",
        {"grade": GRADE, "term": TERMS, "result": SCHOOL_RESULT_OPTS},
        lookup_sql="SELECT code, meaning FROM result_codes"),
    Template("school", "avg_score_dept_credits", 3, "number",
        ["What was the average score in {term} for {dept} courses worth {credits} credits?",
         "In {term}, what was the mean score across {dept} courses with {credits} credits?"],
        f"SELECT AVG(e.score) FROM enrollments e {E_JOIN_C} WHERE c.dept_code = '{{dept}}' AND e.term = '{{term}}' AND c.credits = {{credits}}",
        {"dept": SCHOOL_DEPT_OPTS, "term": TERMS, "credits": CREDITS},
        lookup_sql="SELECT DISTINCT dept_code FROM courses"),
    Template("school", "students_scholarship_grade", 3, "number",
        ["How many grade {grade} students with {scholar} enrolled {period}?",
         "Count the students in grade {grade} who have {scholar} and enrolled {period}."],
        f"SELECT COUNT(*) FROM students WHERE grade_level = {{grade}} AND scholarship_code = '{{scholar}}' AND {when('enrol_date')}",
        {"grade": GRADE, "scholar": SCHOOL_SCHOLAR_OPTS, "period": SCHOOL_PERIODS},
        lookup_sql="SELECT DISTINCT scholarship_code FROM students"),
    Template("school", "top_course_dept_term", 3, "text",
        ["Which {dept} course had the highest average score in {term}?",
         "Which course in the {dept} department scored the highest on average in {term}?"],
        _top_course_dept, {"dept": SCHOOL_DEPT_OPTS, "term": TERMS},
        lookup_sql="SELECT DISTINCT dept_code FROM courses", tie_sql=_top_course_dept_tie),
]

# ---------------------------------------------------------------------------
# Templates: saas (held out for the OOD test)
# ---------------------------------------------------------------------------

SAAS_SIGNUP = year_opts([2022, 2023, 2024]) + half_opts([2022, 2023, 2024])
T_JOIN_A = "JOIN accounts a ON a.account_id = t.account_id"
I_JOIN_A = "JOIN accounts a ON a.account_id = i.account_id"

SAAS_TSTATUS_OPTS = [
    opt("OPN", ["still open", "not yet picked up"]),
    opt("ESC_ENG", ["escalated to the engineering team", "passed on to engineering"]),
    opt("RSV_FIX", ["resolved with a fix", "closed with a working fix"]),
    opt("RSV_WNF", ["closed as won't fix", "declined without a fix"]),
    opt("RSV_DUP", ["closed as duplicates", "merged into another ticket"]),
]
SAAS_PLAN_OPTS = [
    opt("PLN_PRO", ["Pro", "the Pro tier"]),
    opt("PLN_BIZ", ["Business", "the Business tier"]),
    opt("PLN_ENT", ["Enterprise", "the Enterprise tier"]),
    opt("PLN_FR", ["Free", "the free tier"]),
]
SAAS_SEVERITY_OPTS = [
    opt("S1", ["critical outage", "service-down"]),
    opt("S2", ["major degradation", "serious slowdown"]),
    opt("S3", ["minor issue", "small defect"]),
    opt("S4", ["cosmetic or how-to question", "low-priority question"]),
]
SAAS_STATE_OPTS = [
    opt("PD", ["paid", "settled"]),
    opt("OVD", ["overdue", "past due"]),
    opt("VOID", ["voided", "cancelled before payment"]),
    opt("RFD", ["refunded", "paid back to the customer"]),
]


def code_book(kind: str) -> str:
    return f"SELECT code, meaning FROM code_book WHERE kind = '{kind}'"


_top_company, _top_company_tie = top1(
    "a.company", "a.account_id",
    f"FROM invoices i {I_JOIN_A} WHERE a.country = '{{country}}' AND {when('i.invoice_date')}", "SUM(i.amount)",
)
_top_country_state, _top_country_state_tie = top1(
    "a.country", "a.country",
    f"FROM invoices i {I_JOIN_A} WHERE i.state_code = '{{state}}' AND {when('i.invoice_date')}", "SUM(i.amount)",
)

SAAS = [
    Template("saas", "accounts_country_signup", 1, "number",
        ["How many accounts from {country} signed up {period}?",
         "Count the accounts based in {country} created {period}."],
        f"SELECT COUNT(*) FROM accounts WHERE country = '{{country}}' AND {when('signup_date')}",
        {"country": COUNTRY, "period": SAAS_SIGNUP}),
    Template("saas", "invoiced_period_min", 1, "number",
        ["What was the total invoiced {period} counting only invoices of at least {amt}?",
         "What was the sum of invoices {period} that were {amt} or more?"],
        f"SELECT SUM(amount) FROM invoices WHERE {when('invoice_date')} AND amount >= {{amt}}",
        {"period": PERIODS, "amt": nums(100, 250, 500, 1000, 2000)}),
    Template("saas", "tickets_slow_period", 1, "number",
        ["How many tickets opened {period} took more than {h} hours to resolve?",
         "Count the tickets opened {period} with a resolution time above {h} hours."],
        f"SELECT COUNT(*) FROM tickets WHERE {when('opened_date')} AND hours_to_resolve > {{h}}",
        {"period": PERIODS, "h": nums(4, 8, 24, 48, 72)}),
    Template("saas", "billed_country_period", 2, "number",
        ["What was the total amount invoiced to accounts in {country} {period}?",
         "How much did accounts in {country} get invoiced in total {period}?"],
        f"SELECT SUM(i.amount) FROM invoices i {I_JOIN_A} WHERE a.country = '{{country}}' AND {when('i.invoice_date')}",
        {"country": COUNTRY, "period": COARSE}),
    Template("saas", "top_company_invoiced", 2, "text",
        ["Which company in {country} was invoiced the most {period}?",
         "Which {country} account received the largest total of invoices {period}?"],
        _top_company, {"country": COUNTRY, "period": COARSE}, tie_sql=_top_company_tie),
    Template("saas", "avg_hours_country", 2, "number",
        ["What was the average number of hours to resolve a ticket for accounts in {country} opened {period}?",
         "For accounts in {country}, how many hours on average did tickets opened {period} take to resolve?"],
        f"SELECT AVG(t.hours_to_resolve) FROM tickets t {T_JOIN_A} WHERE a.country = '{{country}}' AND {when('t.opened_date')}",
        {"country": COUNTRY, "period": COARSE}),
    Template("saas", "tickets_status_period", 3, "number",
        ["How many tickets opened {period} were {status}?",
         "Count the tickets opened {period} that were {status}."],
        f"SELECT COUNT(*) FROM tickets WHERE status_code = '{{status}}' AND {when('opened_date')}",
        {"status": SAAS_TSTATUS_OPTS, "period": PERIODS}, lookup_sql=code_book("ticket_status")),
    Template("saas", "invoiced_plan_period", 3, "number",
        ["What was the total invoiced {period} to accounts on the {plan} plan?",
         "How much did accounts on {plan} get invoiced in total {period}?"],
        f"SELECT SUM(i.amount) FROM invoices i {I_JOIN_A} WHERE a.plan_code = '{{plan}}' AND {when('i.invoice_date')}",
        {"plan": SAAS_PLAN_OPTS[:3], "period": PERIODS}, lookup_sql=code_book("plan")),
    Template("saas", "avg_hours_severity", 3, "number",
        ["What was the average resolution time in hours for {severity} tickets opened {period}?",
         "On average, how many hours did {severity} tickets opened {period} take to resolve?"],
        f"SELECT AVG(hours_to_resolve) FROM tickets WHERE severity_code = '{{severity}}' AND {when('opened_date')}",
        {"severity": SAAS_SEVERITY_OPTS, "period": PERIODS}, lookup_sql=code_book("severity")),
    Template("saas", "top_country_invoice_state", 3, "text",
        ["Which country had the largest total of {state} invoices {period}?",
         "Accounts in which country had the most money in {state} invoices {period}?"],
        _top_country_state, {"state": SAAS_STATE_OPTS, "period": PERIODS},
        lookup_sql=code_book("invoice_state"), tie_sql=_top_country_state_tie),
]

TEMPLATES: dict[str, list[Template]] = {
    "retail": RETAIL,
    "clinic": CLINIC,
    "logistics": LOGISTICS,
    "school": SCHOOL,
    "saas": SAAS,
}

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _fill(text: str, values: dict[str, str]) -> str:
    return text.format(**values)


def _scalar(result, what: str):
    """The single value of a result, or None if the result is empty or NULL."""
    if not result.ok:
        raise RuntimeError(f"oracle query failed ({what}): {result.error}")
    if result.n_rows == 0:
        return None
    if result.n_rows != 1 or len(result.rows[0]) != 1:
        raise RuntimeError(f"gold query must return one scalar ({what}), got {result.n_rows} rows")
    return result.rows[0][0]


def _has_tie(tpl: Template, sql_values: dict[str, str], db_dir) -> bool:
    res = execute_metered(tpl.domain, _fill(tpl.tie_sql, sql_values), db_dir)
    if not res.ok:
        raise RuntimeError(f"tie query failed for {tpl.name}: {res.error}")
    if res.n_rows < 2:
        return False
    return round(res.rows[0][0], 6) == round(res.rows[1][0], 6)


def _build_task(tpl: Template, combo: tuple[Opt, ...], rng: random.Random, db_dir) -> Task | None:
    """Make one task from one option combination, or None if it is not usable."""
    sql_values, word_values = {}, {}
    for name, option in zip(tpl.params, combo):
        sql_values[name] = option.value
        for key, val in option.extra:
            sql_values[f"{name}_{key}"] = val
        word_values[name] = rng.choice(option.words)
    if "period" in word_values:
        word_values["period_cap"] = word_values["period"][0].upper() + word_values["period"][1:]
    question = _fill(rng.choice(tpl.questions), word_values)
    gold_sql = _fill(tpl.sql, sql_values)

    gold = execute_metered(tpl.domain, gold_sql, db_dir)
    value = _scalar(gold, tpl.name)
    if value is None:
        return None
    if tpl.answer_type == "number":
        if not isinstance(value, (int, float)) or value == 0:
            return None  # zero counts are too easy to guess
        answer = format_number(value)
    else:
        if not isinstance(value, str) or not value.strip():
            return None
        if tpl.tie_sql and _has_tie(tpl, sql_values, db_dir):
            return None
        answer = value

    calls = [{"name": "get_schema", "arguments": {"database": tpl.domain}}]
    rows, vm_steps = 0, gold.vm_steps
    rows += gold.n_rows
    if tpl.lookup_sql:
        lookup = execute_metered(tpl.domain, tpl.lookup_sql, db_dir)
        if not lookup.ok:
            raise RuntimeError(f"oracle lookup failed for {tpl.name}: {lookup.error}")
        calls.append({"name": "run_sql", "arguments": {"database": tpl.domain, "query": tpl.lookup_sql}})
        rows += lookup.n_rows
        vm_steps += lookup.vm_steps
    calls.append({"name": "run_sql", "arguments": {"database": tpl.domain, "query": gold_sql}})
    return Task(
        id="", domain=tpl.domain, level=tpl.level, template=tpl.name, question=question,
        answer_type=tpl.answer_type, gold_sql=gold_sql, gold_answer=answer, oracle_calls=calls,
        optimal_tool_calls=len(calls), optimal_rows=rows, optimal_vm_steps=vm_steps, split="",
    )


def _candidates(tpl: Template, rng: random.Random, seen_questions: set[str], db_dir):
    """Yield usable tasks for one template, in random option order."""
    combos = list(itertools.product(*tpl.params.values()))
    rng.shuffle(combos)
    for combo in combos:
        task = _build_task(tpl, combo, rng, db_dir)
        if task is None or task.question in seen_questions:
            continue
        seen_questions.add(task.question)
        yield task


# One cycle of 14 gives 10 train, 1 val, 3 test_id. 140 tasks = exactly 100/10/30.
_SPLIT_CYCLE = ["val", "test_id", "train", "train", "train", "test_id", "train", "train", "train", "test_id", "train", "train", "train", "train"]


def _generate_domain(domain: str, seed: int, db_dir, seen_questions: set[str]) -> list[Task]:
    rng = random.Random(f"tasks:{seed}:{domain}")
    target = OOD_DOMAIN_TASKS if domain in OOD_DOMAINS else TRAIN_DOMAIN_TASKS
    templates = TEMPLATES[domain]
    streams = [(t, _candidates(t, rng, seen_questions, db_dir)) for t in templates]
    produced: dict[str, list[Task]] = {t.name: [] for t in templates}
    total = 0
    # Round robin over templates so each one gets a similar share.
    while streams and total < target:
        for stream in list(streams):
            if total >= target:
                break
            try:
                produced[stream[0].name].append(next(stream[1]))
                total += 1
            except StopIteration:
                streams.remove(stream)

    tasks, counters = [], {1: 0, 2: 0, 3: 0}
    for tpl in templates:
        for task in produced[tpl.name]:
            counters[task.level] += 1
            task.id = f"{domain}-L{task.level}-{counters[task.level]:04d}"
            tasks.append(task)
    for i, task in enumerate(tasks):
        task.split = "test_ood" if domain in OOD_DOMAINS else _SPLIT_CYCLE[i % len(_SPLIT_CYCLE)]
    return tasks


def generate_tasks(seed: int = 0, db_dir: str | Path | None = None) -> dict[str, list[Task]]:
    """Build all tasks. Returns {split: [Task, ...]}. Same seed gives the same tasks.

    Missing databases are built with the same seed. Existing ones are used as is,
    so pass a fresh `db_dir` (or run gen_data.py) when you change the seed.
    """
    ensure_dbs(db_dir, seed=seed)
    seen_questions: set[str] = set()  # no question text may appear twice, so no split shares one
    by_split: dict[str, list[Task]] = {s: [] for s in SPLITS}
    for domain in DOMAINS:
        for task in _generate_domain(domain, seed, db_dir, seen_questions):
            by_split[task.split].append(task)
    for tasks in by_split.values():
        tasks.sort(key=lambda t: t.id)
    return by_split


def save_tasks(tasks: list[Task], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def load_tasks(path: str | Path) -> list[Task]:
    with Path(path).open(encoding="utf-8") as f:
        return [Task(**json.loads(line)) for line in f if line.strip()]
