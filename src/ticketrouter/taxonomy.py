"""Intent taxonomy for Corvane, a made-up B2B analytics platform.

Each intent has an owning team, a base priority, and the phrasings the synthetic generator uses.
`plain` phrasings carry the obvious keywords. `subtle` ones describe the same need without them,
which is what makes the rules tier miss and the classical tier unsure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRIORITIES = ("urgent", "high", "normal", "low")
HUMAN_QUEUES = {"security_concern": "security_oncall", "outage_report": "incident_response"}
SAFETY_INTENTS = frozenset(HUMAN_QUEUES)


@dataclass(frozen=True)
class Intent:
    name: str
    team: str
    priority: str
    plain: tuple[str, ...]
    subtle: tuple[str, ...]
    spanish: tuple[str, ...] = ()
    confusable: tuple[str, ...] = field(default=())


I = Intent  # noqa: E741 - short alias keeps the table readable

INTENTS: tuple[Intent, ...] = (
    I(
        "billing_dispute",
        "billing",
        "high",
        (
            "We were billed {amount} this month but our contract says {amount2}. Please explain the charge.",
            "Our invoice {invoice} shows {n} seats but we only have {n2} active users. This charge is wrong.",
            "I want to dispute the overage fee on invoice {invoice}.",
            "Why was our card charged twice for the {plan} plan?",
        ),
        (
            "The number on this month's statement does not match what finance approved.",
            "Someone on your side made a mistake with what we owe, the total jumped for no reason.",
        ),
        (
            "Nos cobraron {amount} pero el contrato dice {amount2}. Por favor revisen la factura {invoice}.",
            "Quiero disputar un cargo duplicado en nuestra tarjeta.",
        ),
        ("refund_request", "payment_failed"),
    ),
    I(
        "refund_request",
        "billing",
        "normal",
        (
            "Please refund the {amount} charge for the seats we removed on {date}.",
            "Can we get a refund for the unused months of our annual {plan} plan?",
            "Requesting a refund, we were charged after we downgraded.",
            "I need the money back for invoice {invoice}, we never used the add-on.",
        ),
        (
            "We paid for {n} seats nobody used. Can that come back to us as credit or cash?",
            "Is it possible to get reimbursed for the months the account sat idle?",
        ),
        (
            "Solicito un reembolso de {amount} por los usuarios que eliminamos.",
            "Necesitamos la devolucion del pago de la factura {invoice}.",
        ),
        ("billing_dispute", "cancellation"),
    ),
    I(
        "invoice_copy",
        "billing",
        "low",
        (
            "Can you send a copy of invoice {invoice}? Our accounting team needs it.",
            "Where can I download past invoices for {year}?",
            "Please add our VAT number to the invoice and resend it.",
        ),
        (
            "Finance is asking for the paperwork for last quarter's payments.",
            "Our auditors need the receipts for everything we paid in {year}.",
        ),
        ("Necesito una copia de la factura {invoice} para contabilidad.",),
        ("billing_dispute",),
    ),
    I(
        "payment_failed",
        "billing",
        "high",
        (
            "Our payment failed and the account shows a past due banner.",
            "The card on file was declined, how do I update the payment method?",
            "Got an email that the renewal payment did not go through.",
        ),
        (
            "The bank says they never saw your request, and now the app says we are overdue.",
            "We switched corporate cards and now your system keeps nagging us about money.",
        ),
        ("El pago fallo y la cuenta aparece vencida. Como actualizo la tarjeta?",),
        ("billing_dispute",),
    ),
    I(
        "plan_upgrade",
        "sales",
        "normal",
        (
            "We want to upgrade from {plan} to Enterprise. Who can send a quote?",
            "How much would it cost to add {n} more seats?",
            "Interested in the Enterprise tier for SSO and audit logs, can we talk pricing?",
        ),
        (
            "We are growing fast and will outgrow our current package next quarter. What are our options?",
            "Leadership wants a bigger contract with the extra compliance bits.",
        ),
        ("Queremos pasar al plan Enterprise, pueden enviar una cotizacion?",),
        ("feature_request",),
    ),
    I(
        "cancellation",
        "retention",
        "high",
        (
            "Please cancel our subscription at the end of the term.",
            "We want to close our account and stop the auto renewal.",
            "How do I cancel? We are moving to another tool.",
            "Cancel the {plan} plan effective {date}.",
        ),
        (
            "We have decided not to continue with Corvane after this contract ends.",
            "The team has stopped using the product, we would like to wind things down.",
        ),
        ("Queremos cancelar la suscripcion al final del periodo.",),
        ("refund_request", "data_export"),
    ),
    I(
        "password_reset",
        "account",
        "normal",
        (
            "I forgot my password and the reset email never arrives.",
            "Password reset link says expired every time I click it.",
            "Need to reset my password, I am locked out of {area}.",
            "pw reset pls",
        ),
        (
            "The email with the link to get back in never shows up in my inbox.",
            "I cannot remember my login and the recovery flow loops back to the start.",
        ),
        (
            "Olvide mi contrasena y el correo de recuperacion no llega.",
            "El enlace para restablecer la contrasena dice que expiro.",
        ),
        ("account_locked", "sso_setup"),
    ),
    I(
        "account_locked",
        "account",
        "high",
        (
            "My account is locked after too many login attempts.",
            "User {email} was locked out and cannot sign in.",
            "Account suspended message when I log in, what happened?",
        ),
        (
            "It keeps telling me I have tried too many times and to wait, but I have waited hours.",
            "Our admin can no longer get into the workspace, it says access is blocked.",
        ),
        ("Mi cuenta esta bloqueada despues de varios intentos.",),
        ("password_reset", "security_concern"),
    ),
    I(
        "sso_setup",
        "identity",
        "normal",
        (
            "We need help setting up SAML SSO with our identity provider.",
            "SSO login fails with an invalid assertion error.",
            "How do we enforce single sign-on for all users?",
            "The SAML metadata URL returns 404 during SSO configuration.",
        ),
        (
            "Our IT team wants everyone to log in through the company portal instead of passwords.",
            "Logins through our corporate directory bounce back with a certificate complaint.",
        ),
        ("Necesitamos ayuda para configurar SSO con SAML.",),
        ("user_provisioning", "password_reset"),
    ),
    I(
        "user_provisioning",
        "identity",
        "normal",
        (
            "SCIM provisioning is not creating new users from our directory.",
            "How do I add {n} users in bulk and assign roles?",
            "Deprovisioned users still show up as active seats.",
        ),
        (
            "People who left the company months ago still appear in the workspace.",
            "New hires do not get access automatically like they did last month.",
        ),
        ("Como agrego {n} usuarios y les asigno roles?",),
        ("sso_setup", "billing_dispute"),
    ),
    I(
        "api_rate_limit",
        "platform",
        "high",
        (
            "We keep hitting 429 Too Many Requests on the {endpoint} endpoint.",
            "What is the API rate limit for the {plan} plan? Our sync job gets throttled.",
            "Rate limit errors started today even though our volume did not change.",
            "Can you raise our API quota? We are seeing 429s.",
        ),
        (
            "Our nightly job gets told to slow down halfway through and the run never finishes.",
            "The integration backs off and retries forever once it pulls more than a few thousand records.",
        ),
        ("Recibimos errores 429 en la API, pueden subir el limite?",),
        ("performance_slow", "api_key_issue"),
    ),
    I(
        "api_key_issue",
        "platform",
        "high",
        (
            "Our API key returns 401 Unauthorized since this morning.",
            "How do I rotate an API key without downtime?",
            "The new API token does not have permission to read {area}.",
        ),
        (
            "Our scripts suddenly say the credentials are not valid, nothing changed on our side.",
            "I made a new access credential and it can only see half the data.",
        ),
        ("La clave de API devuelve 401 desde esta manana.",),
        ("api_rate_limit", "security_concern"),
    ),
    I(
        "webhook_failure",
        "platform",
        "high",
        (
            "Webhooks to our endpoint stopped firing on {date}.",
            "Webhook deliveries fail with a timeout and are not retried.",
            "We get duplicate webhook events for the same record.",
        ),
        (
            "Our system stopped hearing about new records from you, nothing arrives on our side.",
            "Notifications to our server show up twice, sometimes three times.",
        ),
        (),
        ("bug_integrations",),
    ),
    I(
        "bug_dashboards",
        "eng_dashboards",
        "high",
        (
            "Dashboard charts show no data after the last release.",
            "The {chart} widget on the dashboard displays the wrong totals.",
            "Dashboard filters reset every time I refresh the page.",
            "Bug: dashboard tiles render blank in {browser}.",
        ),
        (
            "The numbers on the main screen do not add up compared to last week.",
            "The graphs on the home view are empty but the raw tables have rows.",
        ),
        ("Los graficos del dashboard no muestran datos.",),
        ("bug_reporting", "performance_slow"),
    ),
    I(
        "bug_integrations",
        "eng_integrations",
        "high",
        (
            "The CRM connector stopped syncing contacts.",
            "Warehouse sync integration fails with a schema mismatch error.",
            "The chat connector posts alerts to the wrong channel.",
            "Integration error: field mapping drops custom fields.",
        ),
        (
            "Records we add in our sales system no longer show up on your side.",
            "The nightly copy into our data warehouse silently skips some columns.",
        ),
        ("El conector del CRM dejo de sincronizar contactos.",),
        ("webhook_failure", "data_export"),
    ),
    I(
        "bug_mobile",
        "eng_mobile",
        "normal",
        (
            "The mobile app crashes on launch after the update.",
            "iOS app shows a blank screen on the reports tab.",
            "Android app push notifications stopped working.",
        ),
        (
            "On my phone the app closes as soon as I open it.",
            "The tablet version freezes when I rotate the screen.",
        ),
        ("La aplicacion movil se cierra al abrirla.",),
        ("bug_dashboards",),
    ),
    I(
        "bug_reporting",
        "eng_reporting",
        "normal",
        (
            "Scheduled reports are not being emailed anymore.",
            "The PDF report export cuts off the last page.",
            "Report builder throws an error when I add a second grouping.",
        ),
        (
            "The weekly summary that lands in my inbox every Monday did not come.",
            "When I save the monthly summary as a document, half the table is missing.",
        ),
        ("Los reportes programados ya no llegan por correo.",),
        ("bug_dashboards", "data_export"),
    ),
    I(
        "performance_slow",
        "platform",
        "high",
        (
            "The app is very slow today, pages take {n} seconds to load.",
            "Queries time out on large workspaces, performance got worse this week.",
            "Everything is sluggish since the last release.",
        ),
        (
            "It takes forever to open anything and the spinner just keeps going.",
            "We can work but each click waits a long time.",
        ),
        ("La plataforma esta muy lenta hoy.",),
        ("outage_report", "api_rate_limit"),
    ),
    I(
        "feature_request",
        "product",
        "low",
        (
            "Feature request: please add dark mode to the dashboard.",
            "It would be great if we could schedule exports to S3 compatible storage.",
            "Can you add support for custom fiscal calendars?",
            "Suggestion: let us pin favorite reports.",
        ),
        (
            "It would save us hours if the product could group results by region out of the box.",
            "Our team keeps wishing there was a way to comment directly on a chart.",
        ),
        ("Seria genial poder exportar a nuestro propio almacenamiento.",),
        ("plan_upgrade", "how_to_question"),
    ),
    I(
        "data_export",
        "data",
        "normal",
        (
            "How do I export all our data to CSV?",
            "We need a full data export before our contract ends.",
            "The bulk export job has been stuck at {pct} percent for hours.",
        ),
        (
            "We want a copy of everything we have stored with you, in a format we can open.",
            "Before we migrate we need all historical records handed over.",
        ),
        ("Necesitamos exportar todos nuestros datos a CSV.",),
        ("bug_reporting", "cancellation", "data_deletion"),
    ),
    I(
        "data_deletion",
        "privacy",
        "high",
        (
            "Please delete all personal data for {email} under GDPR.",
            "We received a data subject request, how do we erase a user's records?",
            "Request to permanently delete our workspace data.",
        ),
        (
            "A former customer asked us to make sure nothing about them is kept anywhere.",
            "Legal wants confirmation that one person's information is fully wiped.",
        ),
        ("Solicitamos borrar los datos personales de {email} segun el RGPD.",),
        ("data_export", "cancellation"),
    ),
    I(
        "security_concern",
        "security_oncall",
        "urgent",
        (
            "We think our account was compromised, there are logins from {country} we do not recognize.",
            "Possible security breach: an unknown admin user was created in our workspace.",
            "One of our API keys was leaked in a public repo, please help us revoke it.",
            "Suspicious login alerts for {email}, someone may have access to our data.",
            "Phishing email pretending to be Corvane asked our staff for passwords.",
        ),
        (
            "Someone changed our admin settings overnight and nobody here did it.",
            "A user we have never heard of downloaded a large amount of customer data.",
        ),
        ("Creemos que nuestra cuenta fue comprometida, hay accesos desconocidos.",),
        ("account_locked", "api_key_issue"),
    ),
    I(
        "outage_report",
        "incident_response",
        "urgent",
        (
            "Corvane is down for our whole team, we get a 503 error on every page.",
            "Outage: nobody in our company can log in or load dashboards.",
            "Is there an outage? The API returns 500 errors for all requests.",
            "Service unavailable since {time}, production is down for us.",
        ),
        (
            "Nothing loads at all for anyone in our office, this is stopping all work.",
            "All of our users across regions see the error page right now.",
        ),
        ("El servicio esta caido para toda la empresa, error 503 en todas las paginas.",),
        ("performance_slow", "bug_dashboards"),
    ),
    I(
        "how_to_question",
        "support",
        "low",
        (
            "How do I share a dashboard with someone outside my team?",
            "What is the difference between a workspace and a project?",
            "How can I change the timezone for scheduled reports?",
            "Quick question, where do I find the audit log?",
        ),
        (
            "Not sure where to look for the setting that controls who sees what.",
            "Is there a guide for setting this up the way most companies do it?",
        ),
        ("Como comparto un dashboard con alguien fuera de mi equipo?",),
        ("onboarding_help", "feature_request"),
    ),
    I(
        "onboarding_help",
        "success",
        "low",
        (
            "We just signed up and need help with onboarding our team.",
            "Can we schedule an onboarding call for our {n} new users?",
            "Looking for training materials for new admins.",
        ),
        (
            "We are new here and a little lost on where to start.",
            "Our new analysts need a walkthrough before they go live next week.",
        ),
        ("Acabamos de contratar y necesitamos ayuda con la capacitacion.",),
        ("how_to_question",),
    ),
)

# The drift intent: a product that launches mid-stream. It is not in the training taxonomy.
DRIFT_INTENT = Intent(
    "pulse_forecasting",
    "eng_pulse",
    "normal",
    (
        "Corvane Pulse forecasts show flat lines for every metric since we turned it on.",
        "The new Pulse anomaly alerts fire every few minutes, how do we tune sensitivity?",
        "Pulse forecasting model training is stuck in queued state.",
        "How do we pick the seasonality setting in Pulse for weekly data?",
        "Pulse predicted revenue confidence bands look far too wide.",
    ),
    (
        "The new forecasting add-on you launched gives the same prediction for every month.",
        "Since enabling the predictive add-on our anomaly emails never stop.",
    ),
    ("Los pronosticos de Pulse muestran una linea plana.",),
    ("bug_dashboards", "feature_request"),
)

INTENT_NAMES: tuple[str, ...] = tuple(i.name for i in INTENTS)
BY_NAME: dict[str, Intent] = {i.name: i for i in INTENTS} | {DRIFT_INTENT.name: DRIFT_INTENT}
TEAM_OF: dict[str, str] = {i.name: i.team for i in BY_NAME.values()}
TEAMS: tuple[str, ...] = tuple(sorted(set(TEAM_OF.values())))


def team_for(intent: str) -> str:
    return TEAM_OF.get(intent, "support")
