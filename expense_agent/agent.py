# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import os
import re
from typing import Any, AsyncGenerator

from google import genai
from google.genai import types
from google.adk.workflow import Edge, Workflow, node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.apps import App, ResumabilityConfig
from pydantic import BaseModel, Field

from expense_agent import config

# Load .env file if present
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(dotenv_path):
    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

# Define schema for LLM structured output
class RiskAssessment(BaseModel):
    has_risk: bool = Field(description="True if there are any risk factors or anomalies identified in the expense report.")
    risk_factors: list[str] = Field(description="List of risk factors or anomalies found, empty if none.")
    assessment_summary: str = Field(description="A brief summary explanation of the risk assessment.")


def parse_expense_event(node_input: Any) -> dict[str, Any]:
    """Parse raw Pub/Sub message data (base64 or plain) and extract expense details."""
    text = ""
    if hasattr(node_input, "parts") and node_input.parts:
        text = node_input.parts[0].text or ""
    elif isinstance(node_input, str):
        text = node_input
    elif isinstance(node_input, dict):
        payload = node_input
        text = ""
    else:
        payload = {}
        text = ""

    if text:
        try:
            payload = json.loads(text)
        except Exception:
            payload = {"data": text}

    data = payload.get("data")
    if data is None:
        data_payload = payload
    elif isinstance(data, str):
        # Attempt base64 decoding
        try:
            decoded = base64.b64decode(data).decode("utf-8")
            data_payload = json.loads(decoded)
        except Exception:
            # Fallback to parsing direct string as JSON
            try:
                data_payload = json.loads(data)
            except Exception:
                data_payload = {"description": data}
    elif isinstance(data, dict):
        data_payload = data
    else:
        data_payload = {"description": str(data)}

    # Extract amount and other fields
    amount_raw = data_payload.get("amount", 0.0)
    try:
        amount = float(amount_raw)
    except ValueError:
        amount = 0.0

    return {
        "amount": amount,
        "submitter": str(data_payload.get("submitter", "Unknown")),
        "category": str(data_payload.get("category", "General")),
        "description": str(data_payload.get("description", "No description")),
        "date": str(data_payload.get("date", "")),
    }


def parse_expense(ctx: Context, node_input: Any) -> Event:
    """Node 1: Extract expense details and route based on the dollar threshold."""
    expense = parse_expense_event(node_input)
    state_delta = {"expense": expense}

    if expense["amount"] < config.THRESHOLD:
        return Event(output=expense, route="auto_approve", state=state_delta)
    else:
        return Event(output=expense, route="review", state=state_delta)


def auto_approve(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Node 2a: Instantly approve expenses under the threshold."""
    expense = node_input
    result = {
        "status": "APPROVED",
        "reason": f"Amount ${expense['amount']:.2f} is under the threshold of ${config.THRESHOLD:.2f}.",
        "approved_by": "System",
        "expense": expense,
    }
    
    content_text = (
        f"✅ **Auto-Approved:** Expense of **${expense['amount']:.2f}** submitted by "
        f"{expense['submitter']} was automatically approved (under ${config.THRESHOLD:.2f} threshold)."
    )
    
    return Event(
        output=result,
        content=types.Content(role="model", parts=[types.Part.from_text(text=content_text)]),
        state={"final_outcome": result}
    )


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Scrubs SSNs and Credit Card numbers from the text and returns categories redacted."""
    redacted = []
    # SSN pattern: XXX-XX-XXXX or XXXXXXXXX
    ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b')
    # Credit Card pattern: XXXX-XXXX-XXXX-XXXX or XXXX XXXX XXXX XXXX or XXXXXXXXXXXXXXXX
    cc_pattern = re.compile(r'\b(?:\d{4}[\s\- ]?){3}\d{4}\b|\b\d{13,19}\b')
    
    scrubbed = text
    if ssn_pattern.search(scrubbed):
        scrubbed = ssn_pattern.sub("[REDACTED SSN]", scrubbed)
        redacted.append("SSN")
        
    if cc_pattern.search(scrubbed):
        scrubbed = cc_pattern.sub("[REDACTED CREDIT CARD]", scrubbed)
        redacted.append("Credit Card")
        
    return scrubbed, redacted


def detect_prompt_injection(text: str) -> bool:
    """Detects common prompt injection keyword phrases in the text."""
    injection_keywords = [
        "ignore all",
        "ignore previous",
        "ignore the above",
        "system prompt",
        "override rules",
        "bypass rules",
        "auto-approve",
        "force approve",
        "do not audit",
        "set status",
        "you are now",
        "instructions",
        "instead of"
    ]
    text_lower = text.lower()
    for kw in injection_keywords:
        if kw in text_lower:
            return True
    return False


def security_checkpoint(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Security Checkpoint Node: Scrubs PII and defends against prompt injection."""
    expense = dict(node_input)
    description = expense.get("description", "")
    
    # 1. Scrub PII
    scrubbed_desc, redacted = scrub_pii(description)
    expense["description"] = scrubbed_desc
    
    state_delta = {
        "expense": expense,
        "redacted_categories": redacted
    }
    
    # 2. Defend against prompt injection
    injection_detected = detect_prompt_injection(scrubbed_desc)
    
    if injection_detected:
        state_delta["security_event"] = True
        content_text = "🚨 **Security Checkpoint:** Potential prompt injection detected! Bypassing automated LLM review."
        return Event(
            output=expense,
            route="injection_detected",
            content=types.Content(role="model", parts=[types.Part.from_text(text=content_text)]),
            state=state_delta
        )
    else:
        state_delta["security_event"] = False
        if redacted:
            content_text = f"🛡️ **Security Checkpoint:** Sensitive PII redacted from description: {', '.join(redacted)}. Proceeding to LLM risk review."
        else:
            content_text = "🛡️ **Security Checkpoint:** Expense description is clean. Proceeding to LLM risk review."
            
        return Event(
            output=expense,
            route="clean",
            content=types.Content(role="model", parts=[types.Part.from_text(text=content_text)]),
            state=state_delta
        )


def risk_review(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Node 2b: Use gemini-3.1-flash-lite to evaluate potential risk factors."""
    expense = node_input
    
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() in ("true", "1")
    client = genai.Client(vertexai=use_vertex)
    
    prompt = f"""
    You are a financial auditor reviewing an expense report for potential risks, fraud, or anomalies.
    Please review the following expense details:
    - Amount: ${expense['amount']:.2f}
    - Submitter: {expense['submitter']}
    - Category: {expense['category']}
    - Description: {expense['description']}
    - Date: {expense['date']}
    
    Note: Sensitive personal data (like credit cards or SSNs) has already been scrubbed and replaced with placeholders like [REDACTED SSN] by the security gateway. Do not flag the presence of these redaction placeholders as security/compliance risks.
    
    Evaluate if there are any risk factors, such as:
    1. Unusually high amounts for the category.
    2. Vague or suspicious descriptions.
    3. Off-hours or weekend dates if applicable.
    4. Category/description mismatch.
    """
    
    try:
        response = client.models.generate_content(
            model=config.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RiskAssessment,
                temperature=0.1,
            )
        )
        assessment_dict = json.loads(response.text)
        assessment = RiskAssessment(**assessment_dict)
    except Exception as e:
        assessment = RiskAssessment(
            has_risk=True,
            risk_factors=[f"Error during risk review: {str(e)}"],
            assessment_summary="Could not complete automated risk review due to a system error."
        )
        
    state_delta = {"risk_assessment": assessment.model_dump()}
    
    risk_list = ", ".join(assessment.risk_factors) if assessment.risk_factors else "None"
    content_text = (
        f"🔍 **Risk Review Completed:**\n"
        f"- **Has Risk:** {assessment.has_risk}\n"
        f"- **Summary:** {assessment.assessment_summary}\n"
        f"- **Risk Factors:** {risk_list}"
    )
    
    return Event(
        output=expense,
        content=types.Content(role="model", parts=[types.Part.from_text(text=content_text)]),
        state=state_delta
    )


@node(rerun_on_resume=True)
async def human_approval(ctx: Context, node_input: dict[str, Any]) -> AsyncGenerator[Event, None]:
    """Node 3: Pause workflow to ask for human approval, then record the outcome."""
    expense = node_input
    risk_assessment = ctx.state.get("risk_assessment", {})
    security_event = ctx.state.get("security_event", False)
    redacted_categories = ctx.state.get("redacted_categories", [])
    
    # Pause and prompt user if decision is not yet in resume inputs
    if not ctx.resume_inputs or "decision" not in ctx.resume_inputs:
        alert_message = f"⚠️ **Approval Required:** Expense of **${expense['amount']:.2f}** submitted by {expense['submitter']} requires human approval.\n\n"
        
        if security_event:
            alert_message += "🚨 **SECURITY ALERT:** Potential prompt injection detected in the description! Bypassing automated LLM risk review.\n\n"
            
        alert_message += (
            f"**Expense Details:**\n"
            f"- Description: {expense['description']}\n"
            f"- Category: {expense['category']}\n"
            f"- Date: {expense['date']}\n\n"
        )
        
        if redacted_categories:
            alert_message += f"🔒 **PII Redaction Active:** Redacted sensitive data categories: {', '.join(redacted_categories)}.\n\n"
            
        if not security_event:
            alert_message += (
                f"**Risk Audit Summary:**\n"
                f"- Risk Detected: {risk_assessment.get('has_risk', False)}\n"
                f"- Summary: {risk_assessment.get('assessment_summary', 'N/A')}\n\n"
            )
            
        alert_message += "Please respond with 'APPROVED' or 'REJECTED' to make a decision."
        
        yield RequestInput(
            interrupt_id="decision",
            message=alert_message
        )
        return
        
    # Process the decision once resumed
    decision_input = ctx.resume_inputs["decision"]
    if isinstance(decision_input, dict):
        decision_raw = (
            decision_input.get("decision")
            or decision_input.get("result")
            or next(iter(decision_input.values()))
        )
    else:
        decision_raw = decision_input
    decision_raw = str(decision_raw).strip().upper()
    status = "APPROVED" if "APPROVE" in decision_raw else "REJECTED"
    
    result = {
        "status": status,
        "reason": f"Decision made by human: {decision_raw}",
        "approved_by": "Human",
        "expense": expense,
        "risk_assessment": risk_assessment,
        "security_event": security_event,
        "redacted_categories": redacted_categories,
    }
    
    content_text = f"👤 **Human Decision:** Expense was **{status}**.\nReason: {result['reason']}"
    yield Event(
        output=result,
        content=types.Content(role="model", parts=[types.Part.from_text(text=content_text)]),
        state={"final_outcome": result}
    )


# Construct the ADK 2.0 Graph Workflow
root_agent = Workflow(
    name="expense_approval_workflow",
    edges=[
        ('START', parse_expense),
        (parse_expense, {
            "auto_approve": auto_approve,
            "review": security_checkpoint
        }),
        (security_checkpoint, {
            "clean": risk_review,
            "injection_detected": human_approval
        }),
        (risk_review, human_approval),
    ],
    description="An ambient agent that automatically approves small expenses and performs security checks, LLM risk review, and human approval."
)

# App container configuration with Resumability enabled
app = App(
    root_agent=root_agent,
    name="expense_agent",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
