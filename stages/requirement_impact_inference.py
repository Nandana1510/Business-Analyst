"""
Domain- and keyword-aware enrichment for ``impact`` (impacted systems / capabilities).

Merges with LLM-extracted impacts: preserves model output first, then adds inferred
labels from lightweight rules. Capped and de-duplicated to stay relevant, not excessive.
"""

from __future__ import annotations

_MAX_IMPACT = 5

# Any phrase present in the normalized haystack (action + domain + requirement) adds its labels.
_PHRASE_RULES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (
        (
            "pause subscription",
            "resume subscription",
            "cancel subscription",
            "subscription pause",
            "meal subscription",
            "skip delivery",
            "subscription plan",
        ),
        ("Subscription", "Billing"),
    ),
    (
        (
            "address change",
            "change address",
            "update address",
            "shipping address",
            "delivery address",
            "update shipping",
            "billing address",
            "home address",
            "mailing address",
        ),
        ("Settings", "Delivery"),
    ),
    (
        (
            "payment",
            "pay invoice",
            "checkout",
            "charge card",
            "refund",
            "payment method",
            "credit card",
        ),
        ("Billing", "Payment Gateway"),
    ),
    (
        ("cart", "shopping cart", "basket", "add to cart", "remove from cart"),
        ("Cart", "Inventory"),
    ),
    (
        ("inventory", "stock level", "out of stock", "sku", "warehouse"),
        ("Inventory", "Catalog"),
    ),
    (
        ("notification", "push notification", "email receipt", "sms alert"),
        ("Notifications", "Customer profile"),
    ),
    (
        ("report", "analytics", "dashboard", "export csv"),
        ("Reporting", "Analytics"),
    ),
    (
        ("login", "sign in", "sign out", "password reset", "sso", "two-factor", "mfa"),
        ("Authentication", "Customer profile"),
    ),
    (
        ("place order", "order status", "track order", "fulfillment"),
        ("Orders", "Fulfillment"),
    ),
]

# Substring match on the **domain** field only (domain-aware; limits spurious matches in long text).
_DOMAIN_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("subscription", ("Subscription", "Billing")),
    ("billing", ("Billing",)),
    ("payment", ("Billing", "Payment Gateway")),
    ("delivery", ("Delivery", "Fulfillment")),
    ("inventory", ("Inventory",)),
    ("catalog", ("Catalog", "Inventory")),
    ("cart", ("Cart", "Inventory")),
    ("settings", ("Settings",)),
    ("profile", ("Customer profile",)),
    ("account", ("Customer profile",)),
    ("admin", ("Administration",)),
    ("authentication", ("Authentication",)),
    ("auth", ("Authentication",)),
    ("order", ("Orders", "Fulfillment")),
]


def _haystack(action: str, domain: str, requirement_text: str) -> str:
    return " ".join((action or "", domain or "", requirement_text or "")).lower()


def _domain_norm(domain: str) -> str:
    return " ".join((domain or "").lower().split())


def _n_impact(s: str) -> str:
    return " ".join((s or "").lower().split())


def _collapse_redundant_impacts(labels: list[str], hay: str) -> list[str]:
    """
    Merge overlapping labels (e.g. Subscription + Billing + Subscription billing) into
    one clearer name when two or more clearly repeat the same theme.
    """
    lows = [x.strip() for x in labels if x.strip()]
    if len(lows) <= 1:
        return lows
    hay_l = hay.lower()

    # Subscription + billing duplicates (only when multiple cluster labels appear)
    sub_ctx = "subscription" in hay_l or "subscribe" in hay_l or any("subscription" in _n_impact(x) for x in lows)
    if sub_ctx:

        def is_sub_bill_cluster(x: str) -> bool:
            nx = _n_impact(x)
            return nx in ("billing", "subscription", "subscription billing") or (
                "subscription" in nx and "bill" in nx
            )

        cluster = [x for x in lows if is_sub_bill_cluster(x)]
        if len(cluster) >= 2:
            rest = [x for x in lows if x not in cluster]
            lows = list(dict.fromkeys(rest + ["Subscription & billing"]))

    # Delivery / fulfillment duplicates
    if len(lows) >= 2 and any(t in hay_l for t in ("delivery", "shipping", "meal", "fulfill", "track")):

        def is_deliv_cluster(x: str) -> bool:
            nx = _n_impact(x)
            return nx in ("delivery", "meal delivery", "fulfillment", "shipping") or (
                ("deliver" in nx or "fulfill" in nx or "shipping" in nx) and "subscription" not in nx
            )

        cluster = [x for x in lows if is_deliv_cluster(x)]
        if len(cluster) >= 2:
            rest = [x for x in lows if x not in cluster]
            lows = list(dict.fromkeys(rest + ["Delivery & fulfillment"]))

    return lows


def enrich_impact_list(
    action: str,
    domain: str,
    requirement_text: str,
    existing: list[str],
    *,
    max_items: int = _MAX_IMPACT,
) -> list[str]:
    """
    Preserve LLM ``existing`` order, then append phrase- and domain-inferred labels.
    De-duplication is case-insensitive; total length capped at ``max_items``.
    """
    hay = _haystack(action, domain, requirement_text)
    dnorm = _domain_norm(domain)

    seen: set[str] = set()
    out: list[str] = []

    def push(label: str) -> None:
        label = (label or "").strip()
        if not label:
            return
        k = label.lower()
        if k in seen or len(out) >= max_items:
            return
        seen.add(k)
        out.append(label)

    for x in existing:
        push(x)

    for phrases, labels in _PHRASE_RULES:
        if len(out) >= max_items:
            break
        if any(p in hay for p in phrases):
            for lab in labels:
                push(lab)
                if len(out) >= max_items:
                    break

    for key, labels in _DOMAIN_HINTS:
        if len(out) >= max_items:
            break
        if key in dnorm:
            for lab in labels:
                push(lab)
                if len(out) >= max_items:
                    break

    out = _collapse_redundant_impacts(out, hay)[:max_items]
    return out
