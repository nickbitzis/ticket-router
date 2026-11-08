from ticketrouter import rules


def test_clear_password_reset_is_confident():
    r = rules.classify("I forgot my password and the reset email never arrives.")
    assert r.intent == "password_reset" and r.confidence >= 0.9


def test_two_intents_lower_confidence():
    one = rules.classify("Please refund the charge for the seats we removed.")
    two = rules.classify("Please refund the charge. Also we want to cancel our subscription.")
    assert two.confidence < one.confidence


def test_no_match_returns_none():
    r = rules.classify("The numbers on the main screen look odd to me")
    assert r.intent is None and r.confidence == 0.0 and r.matched == []


def test_safety_security_patterns():
    assert rules.safety_check("We think our account was compromised") == "security_concern"
    assert rules.safety_check("Phishing email pretending to be you") == "security_concern"
    assert rules.safety_check("Creemos que nuestra cuenta fue comprometida") == "security_concern"


def test_safety_outage_patterns():
    assert rules.safety_check("Is there an outage? Everything fails") == "outage_report"
    assert rules.safety_check("Corvane is down for our whole team") == "outage_report"
    assert rules.safety_check("El servicio esta caido") == "outage_report"


def test_safety_does_not_fire_on_routine_tickets():
    for text in (
        "How do I export all our data to CSV?",
        "The app is very slow today",
        "Our API key returns 401 Unauthorized",
        "Please cancel our subscription",
    ):
        assert rules.safety_check(text) is None


def test_urgent_words_raise_priority():
    calm = rules.classify("Webhooks to our endpoint stopped firing.")
    hot = rules.classify("Webhooks to our endpoint stopped firing. This is urgent, production is affected.")
    assert calm.priority == "high" and hot.priority == "urgent"


def test_spanish_refund_rule():
    assert rules.classify("Solicito un reembolso de $400").intent == "refund_request"
