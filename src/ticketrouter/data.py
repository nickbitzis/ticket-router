"""Seeded synthetic support tickets.

Every ticket keeps two sets of labels: the true ones the generator used, and the observed ones an
agent would have entered (with a small share of noise). Evaluation scores against observed labels,
the way a real team would, so no tier can reach 100 percent.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

from .taxonomy import BY_NAME, DRIFT_INTENT, INTENT_NAMES, PRIORITIES, SAFETY_INTENTS, team_for

PLANS = ("Starter", "Team", "Business", "Growth")
AREAS = ("the admin console", "reports", "the workspace", "dashboards", "billing settings")
ENDPOINTS = ("/v2/records", "/v2/metrics", "/v2/exports", "/v2/users", "/v2/events")
CHARTS = ("revenue", "funnel", "cohort", "pipeline", "churn")
BROWSERS = ("Chrome", "Firefox", "Safari", "Edge")
COUNTRIES = ("a country we have no staff in", "two unknown regions", "an IP range we do not own")
NAMES = ("Priya", "Marcus", "Elena", "Tomas", "Aiko", "Dev", "Sara", "Jonah", "Lucia", "Omar")
COMPANIES = ("Fernhill Foods", "Atlas Ridge Logistics", "Bluepine Health", "Kestrel Labs", "Marlow & Finch")

OPENERS = ("", "", "Hi team, ", "Hello, ", "Hey support, ", "Good morning. ", "Hi, ")
CLOSERS = ("", "", " Thanks.", " Thank you!", " Regards, {name}.", " Any help appreciated.", " Cheers, {name}")
CONTEXT = (
    " We are on the {plan} plan with about {n} users.",
    " This is for {company}.",
    " Our account manager is out this week so I am writing here.",
    " This started on {date}.",
    " I already checked the help center and did not find an answer.",
    " We have a board meeting on Friday and need this sorted.",
    " Screenshots are attached.",
    " Several people on my team noticed it.",
)
ANGRY = (
    " This is the third time I am asking!!!",
    " Honestly this is unacceptable.",
    " FIX THIS NOW.",
    " We pay a lot for this and it keeps happening.",
    " I am very frustrated.",
)
URGENT = (" This is urgent.", " Please treat as ASAP, production is affected.", " Blocking our whole team.")
# Paraphrase table. Each match is swapped for a random alternative half the time, so the same
# need shows up in many surface forms and the classical tier sees words it never trained on.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "cannot": ("can't", "am unable to", "are not able to", "fail to"),
    "please": ("pls", "kindly", "could you", ""),
    "charged": ("billed", "debited", "invoiced"),
    "charge": ("fee", "line item", "debit"),
    "invoice": ("bill", "statement", "receipt"),
    "account": ("workspace", "org", "tenant", "login"),
    "users": ("people", "members", "staff", "employees"),
    "user": ("member", "person", "employee", "colleague"),
    "stopped": ("quit", "ceased", "no longer started", "broke and stopped"),
    "error": ("failure", "problem", "exception", "glitch"),
    "errors": ("failures", "problems", "exceptions", "glitches"),
    "slow": ("laggy", "sluggish", "crawling", "unusably slow"),
    "help": ("assistance", "support", "a hand", "guidance"),
    "need": ("require", "want", "have to get", "must have"),
    "want": ("would like", "need", "are hoping"),
    "data": ("records", "information", "rows", "content"),
    "team": ("group", "department", "crew", "org"),
    "company": ("business", "organization", "firm"),
    "fails": ("breaks", "errors out", "does not work"),
    "wrong": ("incorrect", "off", "not right", "bogus"),
    "money": ("funds", "cash", "payment"),
    "never": ("still has not", "did not", "never once"),
    "dashboard": ("board", "home view", "main screen"),
    "report": ("summary", "digest", "report pack"),
    "reports": ("summaries", "digests", "report packs"),
    "login": ("sign in", "log on", "access"),
    "settings": ("config", "preferences", "options"),
    "quickly": ("fast", "soon", "right away"),
    "broken": ("busted", "not working", "messed up"),
    "app": ("tool", "platform", "product", "system"),
}
_SYN_RE = re.compile(r"\b(" + "|".join(sorted(SYNONYMS, key=len, reverse=True)) + r")\b")


def paraphrase(text: str, rng: random.Random, rate: float = 0.5) -> str:
    def swap(m: re.Match) -> str:
        if rng.random() >= rate:
            return m.group(0)
        return rng.choice(SYNONYMS[m.group(0)])

    return re.sub(r"\s{2,}", " ", _SYN_RE.sub(swap, text))


SPANISH_OPENERS = ("", "Hola, ", "Buenos dias. ")
SPANISH_CLOSERS = ("", " Gracias.", " Saludos, {name}.")


@dataclass
class Ticket:
    id: str
    text: str
    intent: str  # observed label (what an agent entered)
    priority: str
    team: str
    true_intent: str
    true_priority: str
    language: str
    week: int
    difficulty: float  # 0 easy to 1 hard, drives the simulated LLM tiers
    secondary: str | None = None
    subtle: bool = False
    angry: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _fill(template: str, rng: random.Random) -> str:
    n = rng.choice((3, 5, 8, 12, 20, 25, 40, 60, 150))
    values = {
        "amount": f"${rng.randint(4, 90) * 100:,}",
        "amount2": f"${rng.randint(4, 90) * 100:,}",
        "invoice": f"INV-{rng.randint(10000, 99999)}",
        "n": n,
        "n2": max(1, n - rng.randint(1, max(1, n // 2))),
        "plan": rng.choice(PLANS),
        "date": f"{rng.choice(('March', 'June', 'September', 'October'))} {rng.randint(1, 28)}",
        "year": rng.choice(("2024", "2025")),
        "area": rng.choice(AREAS),
        "email": f"{rng.choice(NAMES).lower()}@{rng.choice(('fernhill', 'kestrel', 'bluepine'))}.example",
        "endpoint": rng.choice(ENDPOINTS),
        "chart": rng.choice(CHARTS),
        "browser": rng.choice(BROWSERS),
        "country": rng.choice(COUNTRIES),
        "pct": rng.choice((12, 40, 73, 99)),
        "time": f"{rng.randint(1, 12)}:{rng.choice(('00', '15', '30', '45'))} {rng.choice(('am', 'pm'))}",
        "name": rng.choice(NAMES),
        "company": rng.choice(COMPANIES),
    }
    return template.format(**values)


def add_typos(text: str, rate: float, rng: random.Random) -> str:
    """Swap, drop, or double a character in a share of the longer words."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 4 and rng.random() < rate:
            j = rng.randrange(1, len(w) - 1)
            op = rng.random()
            if op < 0.4:
                w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
            elif op < 0.75:
                w = w[:j] + w[j + 1 :]
            else:
                w = w[:j] + w[j] + w[j:]
            words[i] = w
    return " ".join(words)


def _shift_priority(p: str, delta: int) -> str:
    i = PRIORITIES.index(p)
    return PRIORITIES[min(len(PRIORITIES) - 1, max(0, i - delta))]


def make_ticket(idx: int, intent_name: str, week: int, rng: random.Random, noise: float = 0.04) -> Ticket:
    intent = BY_NAME[intent_name]
    spanish = bool(intent.spanish) and rng.random() < 0.06
    subtle = (not spanish) and rng.random() < 0.22
    difficulty = 0.0
    if spanish:
        body = _fill(rng.choice(intent.spanish), rng)
        opener, closer = rng.choice(SPANISH_OPENERS), rng.choice(SPANISH_CLOSERS)
        difficulty += 0.25
    else:
        body = _fill(rng.choice(intent.subtle if subtle else intent.plain), rng)
        opener, closer = rng.choice(OPENERS), rng.choice(CLOSERS)
        difficulty += 0.45 if subtle else 0.0

    secondary = None
    if not spanish and rng.random() < 0.18:
        pool = [c for c in intent.confusable if c in INTENT_NAMES] or list(INTENT_NAMES)
        secondary = rng.choice(pool) if rng.random() < 0.7 else rng.choice(INTENT_NAMES)
        if secondary == intent_name:
            secondary = None
        else:
            extra = _fill(rng.choice(BY_NAME[secondary].plain), rng)
            body = f"{body} Also, {extra[0].lower()}{extra[1:]}" if rng.random() < 0.7 else f"{extra} {body}"
            difficulty += 0.35
            if secondary in SAFETY_INTENTS and intent_name not in SAFETY_INTENTS:
                # Labeling policy: a ticket that mentions a breach or an outage is filed under it.
                intent_name, secondary = secondary, intent_name
                intent = BY_NAME[intent_name]

    text = opener + (body if spanish else paraphrase(body, rng))
    long_ticket = rng.random() < 0.25
    if long_ticket and not spanish:
        for c in rng.sample(CONTEXT, k=rng.randint(2, 4)):
            text += _fill(c, rng)
    angry = rng.random() < 0.12
    urgent = rng.random() < 0.1
    if angry and not spanish:
        text += rng.choice(ANGRY)
    if urgent and not spanish:
        text += rng.choice(URGENT)
    text += _fill(closer, rng)

    if rng.random() < 0.2:
        text = add_typos(text, rate=0.25, rng=rng)
        difficulty += 0.15
    if len(text) < 40:
        difficulty += 0.15
    if rng.random() < 0.08:
        text = text.lower()

    true_priority = intent.priority
    if intent.priority != "urgent":
        if urgent or (angry and rng.random() < 0.5):
            true_priority = _shift_priority(true_priority, 1)
        elif long_ticket and rng.random() < 0.1:
            true_priority = _shift_priority(true_priority, -1)

    observed_intent = intent_name
    if rng.random() < noise:
        observed_intent = rng.choice(intent.confusable or INTENT_NAMES)
    observed_priority = true_priority
    if rng.random() < noise:
        observed_priority = _shift_priority(true_priority, rng.choice((-1, 1)))
    team = team_for(observed_intent)

    return Ticket(
        id=f"T{idx:05d}",
        text=text.strip(),
        intent=observed_intent,
        priority=observed_priority,
        team=team,
        true_intent=intent_name,
        true_priority=true_priority,
        language="es" if spanish else "en",
        week=week,
        difficulty=round(min(1.0, difficulty + rng.random() * 0.1), 3),
        secondary=secondary,
        subtle=subtle,
        angry=angry,
    )


# Relative volume per intent. Password resets and how-to questions are common, outages rare.
WEIGHTS = {
    "password_reset": 9,
    "how_to_question": 8,
    "billing_dispute": 5,
    "refund_request": 5,
    "invoice_copy": 4,
    "payment_failed": 4,
    "plan_upgrade": 3,
    "cancellation": 3,
    "account_locked": 5,
    "sso_setup": 4,
    "user_provisioning": 3,
    "api_rate_limit": 4,
    "api_key_issue": 3,
    "webhook_failure": 3,
    "bug_dashboards": 5,
    "bug_integrations": 4,
    "bug_mobile": 3,
    "bug_reporting": 4,
    "performance_slow": 3,
    "feature_request": 5,
    "data_export": 3,
    "data_deletion": 2,
    "security_concern": 2,
    "outage_report": 1.5,
    "onboarding_help": 3,
}


def generate(n: int = 20_000, seed: int = 7, weeks: int = 12, noise: float = 0.04) -> list[Ticket]:
    """Base ticket stream over `weeks` weeks. No drift intent here, see `drift_stream`."""
    rng = random.Random(seed)
    names = list(WEIGHTS)
    weights = [WEIGHTS[k] for k in names]
    tickets = []
    for i in range(n):
        intent = rng.choices(names, weights=weights)[0]
        tickets.append(make_ticket(i, intent, week=i * weeks // n, rng=rng, noise=noise))
    return tickets


def partition(tickets: list[Ticket], train_share: float = 0.2, test_share: float = 0.3, seed: int = 7):
    """Labeled training set, held-out test set, and the rest as an unlabeled incoming pool.

    With 20,000 tickets that is 4,000 labeled, 6,000 test, 10,000 pool. A small labeled seed is the
    realistic starting point: the pool is what the feedback loop and the drift monitor consume.
    """
    rng = random.Random(seed + 1)
    idx = list(range(len(tickets)))
    rng.shuffle(idx)
    a = int(len(idx) * train_share)
    b = a + int(len(idx) * test_share)
    pick = lambda ids: [tickets[i] for i in sorted(ids)]  # noqa: E731
    return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])


def make_drift_ticket(idx: int, week: int, rng: random.Random) -> Ticket:
    """A ticket about Corvane Pulse, a product launched mid-stream that the models never saw."""
    return make_ticket(idx, DRIFT_INTENT.name, week, rng, noise=0.0)


def drift_stream(
    base: list[Ticket],
    per_week: int = 500,
    weeks: int = 12,
    launch_week: int = 7,
    share_by_week: tuple[float, ...] = (0.06, 0.1, 0.14, 0.18, 0.2),
    seed: int = 11,
) -> list[Ticket]:
    """Weekly stream of known tickets from `base`, with the drift intent mixed in from `launch_week`."""
    rng = random.Random(seed)
    pool = list(base)
    rng.shuffle(pool)
    out: list[Ticket] = []
    next_id = 90_000
    k = 0
    for w in range(weeks):
        share = 0.0
        if w >= launch_week:
            share = share_by_week[min(w - launch_week, len(share_by_week) - 1)]
        n_new = int(per_week * share)
        for _ in range(per_week - n_new):
            t = pool[k % len(pool)]
            k += 1
            out.append(Ticket(**{**t.to_dict(), "week": w}))
        for _ in range(n_new):
            out.append(make_drift_ticket(next_id, w, rng))
            next_id += 1
    return out
