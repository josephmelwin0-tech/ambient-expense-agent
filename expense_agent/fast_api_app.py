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
import io
import json
import logging
import os
import sys
from contextlib import aclosing
from typing import Any, Dict, Optional

import google.auth
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel, Field

from expense_agent.app_utils.telemetry import setup_telemetry
from expense_agent.app_utils.typing import Feedback

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

# Configure standard Python logging for console logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("ambient_expense_service")

# Load the expense approval agent app
from expense_agent.agent import app as agent_app

setup_telemetry()
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")
AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
session_service_uri = None
artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

# Initialize App using standard ADK Fast API app generator with otel_to_cloud=False
app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=False,  # Set otel_to_cloud=False as requested
)
app.title = "ambient-expense-agent"
app.description = "API for interacting with the Agent ambient-expense-agent"

# Dedicated InMemoryRunner for our custom Pub/Sub endpoints
runner = InMemoryRunner(app=agent_app)

# Pub/Sub Payload Pydantic Models
class PubSubMessage(BaseModel):
    data: Optional[str] = Field(None, description="Base64 encoded payload or raw string")
    attributes: Optional[Dict[str, Any]] = None
    messageId: Optional[str] = None
    publishTime: Optional[str] = None

class PubSubPushRequest(BaseModel):
    message: PubSubMessage
    subscription: str


def serialize_event(event) -> dict:
    """Safely converts an ADK Event object to a JSON-serializable dictionary."""
    try:
        return jsonable_encoder(event)
    except Exception as e:
        logger.error("Failed to serialize event: %s", e)
        return {"error": f"Serialization error: {str(e)}"}


async def ensure_session_exists(session_id: str):
    """Checks if session exists on the runner, and creates it if not."""
    app_name = agent_app.name or "expense_agent"
    user_id = "default-user"
    try:
        session = await runner.session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
        if session is None:
            logger.info("Session '%s' not found. Creating it...", session_id)
            await runner.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id
            )
    except Exception:
        logger.info("Session '%s' does not exist. Creating it...", session_id)
        await runner.session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback."""
    return {"status": "success"}


@app.post("/")
async def handle_pubsub_trigger(request: Request, session_id: Optional[str] = None):
    """
    Accepts Pub/Sub push trigger messages, normalizes the subscription path,
    and executes the agent workflow.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        logger.error("Failed to parse request JSON: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Check if this matches a Pub/Sub push message format
    if isinstance(body, dict) and "message" in body and "subscription" in body:
        try:
            push_req = PubSubPushRequest(**body)
            # Normalize subscription path to keep session records readable
            sub_path = push_req.subscription
            normalized_sub_id = sub_path.split("/")[-1]
            session_id = normalized_sub_id
            
            message_payload = push_req.message.model_dump()
            text_input = json.dumps(message_payload)
            logger.info("Normalized Pub/Sub trigger. Sub name: '%s', Session ID: '%s'", normalized_sub_id, session_id)
        except Exception as e:
            logger.warning("Pub/Sub structure present but failed to validate: %s. Falling back to direct parsing.", e)
            normalized_sub_id = "local-trigger"
            session_id = session_id or "local-session"
            text_input = json.dumps(body)
    else:
        # Direct raw payload testing
        logger.info("Direct JSON payload received (not a Pub/Sub envelope).")
        session_id = session_id or "local-session"
        text_input = json.dumps(body)

    await ensure_session_exists(session_id)

    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=text_input)]
    )

    events_emitted = []
    paused_on_interrupt = False
    interrupt_id = None
    interrupt_message = None

    logger.info("Executing workflow for Session ID: '%s'...", session_id)

    try:
        async with aclosing(
            runner.run_async(
                user_id="default-user",
                session_id=session_id,
                new_message=new_message
            )
        ) as agen:
            async for event in agen:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            logger.info("[Event Text] %s", part.text.strip())
                        if part.function_call and part.function_call.name == "adk_request_input":
                            paused_on_interrupt = True
                            fc_args = part.function_call.args
                            interrupt_id = fc_args.get("interruptId")
                            interrupt_message = fc_args.get("message")
                            logger.info("[Workflow Paused] Waiting on human input for interrupt_id='%s'", interrupt_id)

                if event.output:
                    logger.info("[Node Output] %s", event.output)

                events_emitted.append(serialize_event(event))

    except Exception as e:
        logger.exception("Error executing workflow:")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

    response_data = {
        "session_id": session_id,
        "status": "PAUSED" if paused_on_interrupt else "COMPLETED",
        "events": events_emitted
    }
    if paused_on_interrupt:
        response_data["interrupt"] = {
            "id": interrupt_id,
            "prompt": interrupt_message
        }

    return JSONResponse(content=response_data)


@app.post("/sessions/{session_id}/decision")
async def submit_human_decision(session_id: str, payload: dict):
    """
    Submits a human decision (APPROVED / REJECTED) to resume a paused workflow session.
    """
    decision = payload.get("decision", "").strip().upper()
    if not decision:
        raise HTTPException(status_code=400, detail="Missing 'decision' field (must be APPROVED or REJECTED)")

    logger.info("Resuming session '%s' with human decision: %s", session_id, decision)

    new_message = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="adk_request_input",
                    id="decision",
                    response={"decision": decision}
                )
            )
        ]
    )

    events_emitted = []
    try:
        async with aclosing(
            runner.run_async(
                user_id="default-user",
                session_id=session_id,
                new_message=new_message
            )
        ) as agen:
            async for event in agen:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            logger.info("[Event Text] %s", part.text.strip())
                if event.output:
                    logger.info("[Node Output] %s", event.output)
                events_emitted.append(serialize_event(event))
    except Exception as e:
        logger.exception("Error resuming workflow:")
        raise HTTPException(status_code=500, detail=f"Failed to resume session: {str(e)}")

    return JSONResponse(content={
        "session_id": session_id,
        "status": "COMPLETED",
        "events": events_emitted
    })


if __name__ == "__main__":
    uvicorn.run("expense_agent.fast_api_app:app", host="0.0.0.0", port=8080, log_level="info")
