"""
src/alerts/notifier.py
Sends email and Slack alerts when a customer's churn score crosses the threshold.
"""

import smtplib
import requests
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", 587))
SMTP_USER     = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_TO      = os.getenv("ALERT_RECIPIENT")
SLACK_URL     = os.getenv("SLACK_WEBHOOK_URL")
THRESHOLD     = float(os.getenv("CHURN_THRESHOLD", 0.65))


def _build_email_body(customer_id: str, prob: float, reasons: list) -> str:
    reasons_html = "".join(
        f"<li><b>{r['label']}</b> — {r['direction']} (SHAP: {r['shap_value']:.3f})</li>"
        for r in reasons
    )
    return f"""
    <html><body>
    <h2>🔴 ChurnRadar Alert — High Churn Risk Detected</h2>
    <p><b>Customer ID:</b> {customer_id}</p>
    <p><b>Churn Probability:</b> {prob:.1%}</p>
    <h3>Top Churn Drivers:</h3>
    <ul>{reasons_html}</ul>
    <p><b>Recommended Action:</b> Assign to CS team immediately.
       Offer loyalty discount or contract upgrade.</p>
    <hr>
    <small>Sent by ChurnRadar — Retention Intelligence Platform</small>
    </body></html>
    """


def send_email_alert(customer_id: str, prob: float, reasons: list):
    """Send HTML email alert to the CS team."""
    if not all([SMTP_USER, SMTP_PASSWORD, ALERT_TO]):
        print("[WARN] Email credentials not set. Skipping email alert.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🔴 ChurnRadar: High Risk Customer {customer_id} ({prob:.0%})"
    msg["From"]    = SMTP_USER
    msg["To"]      = ALERT_TO
    msg.attach(MIMEText(_build_email_body(customer_id, prob, reasons), "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, ALERT_TO, msg.as_string())
        print(f"[OK] Email alert sent for {customer_id}")
    except Exception as e:
        print(f"[ERROR] Email failed: {e}")


def send_slack_alert(customer_id: str, prob: float, reasons: list):
    """Send a Slack block message to the CS webhook."""
    if not SLACK_URL:
        print("[WARN] Slack webhook not set. Skipping.")
        return

    reason_text = "\n".join(
        f"• *{r['label']}* — {r['direction']}" for r in reasons
    )

    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🔴 ChurnRadar — High Risk Alert"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Customer ID:*\n{customer_id}"},
                    {"type": "mrkdwn", "text": f"*Churn Probability:*\n{prob:.1%}"},
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Top Churn Drivers:*\n{reason_text}"}
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "✅ *Action:* Assign to CS team. Offer loyalty discount or contract upgrade."
                }
            }
        ]
    }

    resp = requests.post(SLACK_URL, json=payload, timeout=5)
    if resp.status_code == 200:
        print(f"[OK] Slack alert sent for {customer_id}")
    else:
        print(f"[ERROR] Slack failed: {resp.status_code} {resp.text}")


def fire_alert(customer_id: str, prob: float, reasons: list):
    """Fire both email and Slack alerts if prob exceeds threshold."""
    if prob >= THRESHOLD:
        print(f"[ALERT] {customer_id} churn prob={prob:.2%} — firing alerts")
        send_email_alert(customer_id, prob, reasons)
        send_slack_alert(customer_id, prob, reasons)
    else:
        print(f"[INFO] {customer_id} churn prob={prob:.2%} — below threshold, no alert")
