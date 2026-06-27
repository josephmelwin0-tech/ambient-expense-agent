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

import os
import re
import json
import logging
from typing import Optional, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import vertexai
from google.adk.sessions import VertexAiSessionService
from google.cloud import aiplatform_v1beta1 as gapic_v1beta1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manager_dashboard")

# Load .env file manually if present
for env_dir in [os.getcwd(), os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
    dotenv_path = os.path.join(env_dir, ".env")
    if os.path.exists(dotenv_path):
        logger.info(f"Loading environment from {dotenv_path}")
        with open(dotenv_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

# Read configurations
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID")
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1")
runtime_id = os.environ.get("AGENT_RUNTIME_ID", "")

# Extract engine_id
engine_id = ""
if runtime_id:
    match = re.search(r"projects/([^/]+)/locations/([^/]+)/reasoningEngines/(\d+)", runtime_id)
    if match:
        extracted_project = match.group(1)
        extracted_location = match.group(2)
        engine_id = match.group(3)
        if not project_id:
            project_id = extracted_project
        location = extracted_location
    else:
        engine_id = runtime_id

# Fallback check from deployment metadata file if present
if not project_id or not engine_id:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_path = os.path.join(parent_dir, "deployment_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                remote_id = meta.get("remote_agent_runtime_id", "")
                if remote_id:
                    match = re.search(r"projects/([^/]+)/locations/([^/]+)/reasoningEngines/(\d+)", remote_id)
                    if match:
                        project_id = match.group(1)
                        location = match.group(2)
                        engine_id = match.group(3)
                        logger.info(f"Loaded config from deployment metadata: Project={project_id}, Engine={engine_id}")
        except Exception as e:
            logger.warning(f"Could not load deployment metadata: {e}")

logger.info(f"Initialization details - Project: {project_id}, Location: {location}, Engine ID: {engine_id}")

if project_id and location:
    vertexai.init(project=project_id, location=location)

from fastapi import status
import secrets

# Initialize FastAPI App
app = FastAPI(
    title="Aether Audit Manager Dashboard",
    description="Sleek, premium manager approval dashboard for the Ambient Expense Agent.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class DecisionAction(BaseModel):
    approved: bool
    interrupt_id: str
    passcode: Optional[str] = None

@app.get("/api/pending")
async def get_pending():
    """Queries VertexAiSessionService to retrieve sessions with unresolved adk_request_input interrupts."""
    if not project_id or not engine_id:
        raise HTTPException(
            status_code=500,
            detail="GCP project ID or AGENT_RUNTIME_ID environment variable is missing. Please set them to start."
        )
    
    try:
        svc = VertexAiSessionService(
            project=project_id,
            location=location,
            agent_engine_id=engine_id
        )
        
        # Enumerate sessions
        list_resp = await svc.list_sessions(app_name="expense_agent")
        pending_approvals = []
        
        for session in list_resp.sessions:
            try:
                # Retrieve the full session with its event history.
                # Must query with session.user_id to satisfy ownership constraint.
                full_session = await svc.get_session(
                    app_name="expense_agent",
                    user_id=session.user_id,
                    session_id=session.id
                )
                if not full_session or not full_session.events:
                    continue
                
                # Scan session events for unresolved adk_request_input calls
                unresolved = {}
                for event in full_session.events:
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.function_call and part.function_call.name == "adk_request_input":
                                fid = part.function_call.id
                                msg = part.function_call.args.get("message", "") if part.function_call.args else ""
                                unresolved[fid] = {
                                    "interrupt_id": fid,
                                    "message": msg,
                                    "timestamp": event.timestamp,
                                }
                            if part.function_response and part.function_response.name == "adk_request_input":
                                fid = part.function_response.id
                                if fid in unresolved:
                                    del unresolved[fid]
                                    
                # Add unresolved interrupts with state payload metadata
                for fid, interrupt in unresolved.items():
                    pending_approvals.append({
                        "session_id": session.id,
                        "user_id": session.user_id,
                        "interrupt_id": fid,
                        "message": interrupt["message"],
                        "timestamp": interrupt["timestamp"],
                        "expense": full_session.state.get("expense"),
                        "risk_assessment": full_session.state.get("risk_assessment"),
                        "security_event": full_session.state.get("security_event", False),
                        "redacted_categories": full_session.state.get("redacted_categories", []),
                        # Formulate historical timeline events for slide-out audit review
                        "events": [
                            {
                                "author": ev.author,
                                "timestamp": ev.timestamp,
                                "content": ev.content.parts[0].text if ev.content and ev.content.parts and ev.content.parts[0].text else None,
                                "actions": ev.actions.state_delta if ev.actions else None,
                                "route": ev.actions.route if ev.actions else None,
                            }
                            for ev in full_session.events
                            if ev.content or (ev.actions and (ev.actions.state_delta or ev.actions.route))
                        ]
                    })
            except Exception as se:
                logger.error(f"Error checking session {session.id}: {se}")
                continue
                
        return pending_approvals
    except Exception as e:
        logger.exception("Failed to query pending sessions")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/action/{session_id}")
async def resume_session(session_id: str, action: DecisionAction):
    """Resumes the paused session on Agent Runtime using low-level GAPIC client call."""
    if not project_id or not engine_id:
        raise HTTPException(
            status_code=500,
            detail="GCP project ID or AGENT_RUNTIME_ID environment variable is missing. Please set them to start."
        )
    
    try:
        # Verify manager passcode
        correct_passcode = os.environ.get("DASHBOARD_PASSWORD", "admin123")
        if not action.passcode or not secrets.compare_digest(action.passcode, correct_passcode):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Manager Passcode. Action rejected."
            )
        
        from vertexai.preview.reasoning_engines import ReasoningEngine
        
        resource_name = f"projects/{project_id}/locations/{location}/reasoningEngines/{engine_id}"
        agent = ReasoningEngine(resource_name)
        
        # Build the exact resume payload required by agent.py human_approval node
        resume_payload = {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": action.interrupt_id,
                        "name": "adk_request_input",
                        "response": {
                            "approved": action.approved,
                            "decision": "APPROVED" if action.approved else "REJECTED"
                        }
                    }
                }
            ]
        }
        
        # Construct low-level GAPIC request. Bypasses client method registration failure due to async mode.
        req = gapic_v1beta1.StreamQueryReasoningEngineRequest(
            name=resource_name,
            input={
                "message": resume_payload,
                "user_id": "default-user",
                "session_id": session_id
            },
            class_method="stream_query"
        )
        
        # Invoke StreamQuery and consume stream iterator to completion
        response_stream = agent.execution_api_client.stream_query_reasoning_engine(request=req)
        
        results = []
        for chunk in response_stream:
            try:
                data_str = chunk.data.decode("utf-8")
                for line in data_str.strip().split("\n"):
                    if line.strip():
                        results.append(json.loads(line))
            except Exception:
                pass
                
        return {"success": True, "results": results}
        
    except Exception as e:
        logger.exception(f"Failed to resume session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the beautiful, premium dark-themed manager approval dashboard HTML page."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aether Audit - Manager Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #07080d;
            --panel: rgba(13, 15, 27, 0.45);
            --panel-hover: rgba(20, 24, 43, 0.55);
            --panel-border: rgba(255, 255, 255, 0.06);
            --panel-border-hover: rgba(255, 255, 255, 0.12);
            --text: #f3f4f6;
            --text-muted: #8c92ac;
            --primary: #6366f1;
            --primary-rgb: 99, 102, 241;
            --primary-glow: rgba(99, 102, 241, 0.12);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.15);
            --danger: #f43f5e;
            --danger-glow: rgba(244, 63, 94, 0.15);
            --warning: #f59e0b;
            --warning-glow: rgba(245, 158, 11, 0.15);
            --purple: #a855f7;
            --font-sans: 'Outfit', sans-serif;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg);
            color: var(--text);
            font-family: var(--font-sans);
            min-height: 100vh;
            overflow-x: hidden;
            background-image: radial-gradient(ellipse at 50% 0%, #151934 0%, #07080d 80%);
            background-attachment: fixed;
        }

        /* Ambient Glow Blobs */
        .ambient-glow {
            position: fixed;
            width: 700px;
            height: 700px;
            border-radius: 50%;
            pointer-events: none;
            z-index: -1;
            filter: blur(160px);
            opacity: 0.15;
            transition: all 1s ease;
        }

        .glow-1 {
            top: -200px;
            left: -150px;
            background: radial-gradient(circle, var(--primary) 0%, transparent 70%);
        }

        .glow-2 {
            bottom: -250px;
            right: -150px;
            background: radial-gradient(circle, var(--success) 0%, transparent 70%);
        }

        /* Header Styling */
        header {
            position: sticky;
            top: 0;
            z-index: 50;
            background: rgba(7, 8, 13, 0.5);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--panel-border);
            padding: 20px 6%;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
        }

        .logo-icon svg {
            width: 20px;
            height: 20px;
            fill: white;
        }

        .logo-text {
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #ffffff, var(--text-muted));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--panel-border);
            padding: 8px 16px;
            border-radius: 30px;
            font-size: 14px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--success);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 0.4; }
            50% { opacity: 1; }
            100% { opacity: 0.4; }
        }

        /* Main Container */
        main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 60px 6%;
        }

        .dashboard-header {
            margin-bottom: 48px;
        }

        .dashboard-title {
            font-size: 38px;
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: 12px;
        }

        .dashboard-subtitle {
            color: var(--text-muted);
            font-size: 16px;
            font-weight: 400;
        }

        /* Grid Layout */
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
            gap: 30px;
        }

        /* Glassmorphic Card */
        .card {
            background: var(--panel);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            padding: 32px;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            gap: 24px;
            transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1),
                        border-color 0.4s ease,
                        box-shadow 0.4s ease;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(800px circle at var(--x, 0px) var(--y, 0px), rgba(255, 255, 255, 0.05), transparent 40%);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.5s ease;
        }

        .card:hover::before {
            opacity: 1;
        }

        .card:hover {
            transform: translateY(-6px);
            border-color: var(--panel-border-hover);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4),
                        0 0 30px rgba(var(--primary-rgb), 0.08);
        }

        /* Flashing warning card state for security events */
        .card.security-threat {
            border-color: rgba(244, 63, 94, 0.3);
            box-shadow: 0 0 20px rgba(244, 63, 94, 0.1);
        }

        .card.security-threat:hover {
            border-color: rgba(244, 63, 94, 0.6);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4),
                        0 0 30px rgba(244, 63, 94, 0.15);
        }

        /* Card Header */
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .category-badge {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 6px 14px;
            border-radius: 30px;
        }

        .badge-meals {
            color: #f97316;
            background: rgba(249, 115, 22, 0.1);
            border: 1px solid rgba(249, 115, 22, 0.2);
        }

        .badge-travel {
            color: #3b82f6;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.2);
        }

        .badge-software {
            color: #a855f7;
            background: rgba(168, 85, 247, 0.1);
            border: 1px solid rgba(168, 85, 247, 0.2);
        }

        .badge-general {
            color: #94a3b8;
            background: rgba(148, 163, 184, 0.1);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }

        .card-date {
            font-size: 14px;
            color: var(--text-muted);
            font-weight: 500;
        }

        /* Card Info */
        .submitter-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .avatar {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4f46e5, var(--primary));
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
            box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);
        }

        .submitter-email {
            font-size: 15px;
            font-weight: 600;
        }

        .expense-desc {
            font-size: 18px;
            font-weight: 500;
            line-height: 1.5;
            color: #ffffff;
        }

        .expense-amount {
            font-size: 36px;
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(to right, #ffffff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Audit Summary Panels */
        .audit-summary-box {
            padding: 16px 20px;
            border-radius: 16px;
            font-size: 13px;
            line-height: 1.5;
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }

        .audit-summary-box svg {
            width: 18px;
            height: 18px;
            flex-shrink: 0;
            margin-top: 2px;
        }

        .audit-risk-high {
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.15);
            color: #f59e0b;
        }

        .audit-risk-clean {
            background: rgba(16, 185, 129, 0.06);
            border: 1px solid rgba(16, 185, 129, 0.12);
            color: var(--success);
        }

        .audit-threat {
            background: rgba(244, 63, 94, 0.08);
            border: 1px solid rgba(244, 63, 94, 0.18);
            color: var(--danger);
            animation: pulse-border 2s infinite;
        }

        @keyframes pulse-border {
            0% { border-color: rgba(244, 63, 94, 0.18); }
            50% { border-color: rgba(244, 63, 94, 0.5); }
            100% { border-color: rgba(244, 63, 94, 0.18); }
        }

        /* Action Buttons */
        .card-actions {
            display: flex;
            gap: 14px;
            margin-top: 8px;
        }

        .btn {
            font-family: var(--font-sans);
            font-size: 14px;
            font-weight: 700;
            padding: 14px 24px;
            border-radius: 14px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            border: none;
            outline: none;
            flex: 1;
        }

        .btn-approve {
            background: linear-gradient(135deg, var(--success), #059669);
            color: #ffffff;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25);
        }

        .btn-approve:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
            filter: brightness(1.1);
        }

        .btn-reject {
            background: rgba(244, 63, 94, 0.05);
            border: 1px solid rgba(244, 63, 94, 0.3);
            color: var(--danger);
        }

        .btn-reject:hover:not(:disabled) {
            background: var(--danger);
            color: white;
            border-color: var(--danger);
            box-shadow: 0 6px 20px rgba(244, 63, 94, 0.3);
            transform: translateY(-2px);
        }

        .btn-audit {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--panel-border);
            color: var(--text-muted);
        }

        .btn-audit:hover:not(:disabled) {
            background: rgba(255, 255, 255, 0.08);
            color: white;
            border-color: rgba(255, 255, 255, 0.15);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }

        /* Slide-out Drawer Panel */
        .drawer-overlay {
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(0, 0, 0, 0.6);
            opacity: 0;
            pointer-events: none;
            z-index: 90;
            transition: opacity 0.5s cubic-bezier(0.16, 1, 0.3, 1);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
        }

        .drawer-overlay.open {
            opacity: 1;
            pointer-events: all;
        }

        .drawer {
            position: fixed;
            top: 0;
            right: -500px;
            width: 500px;
            height: 100vh;
            background: rgba(10, 11, 19, 0.96);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            border-left: 1px solid rgba(255, 255, 255, 0.07);
            z-index: 100;
            transition: right 0.5s cubic-bezier(0.16, 1, 0.3, 1);
            padding: 40px;
            box-shadow: -20px 0 50px rgba(0, 0, 0, 0.6);
            display: flex;
            flex-direction: column;
            gap: 32px;
            overflow-y: auto;
        }

        .drawer.open {
            right: 0;
        }

        .drawer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .drawer-title {
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        .close-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--panel-border);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .close-btn:hover {
            background: rgba(255, 255, 255, 0.12);
            color: white;
            transform: rotate(90deg);
        }

        /* Timeline in Drawer */
        .timeline {
            position: relative;
            padding-left: 32px;
            margin-top: 10px;
        }

        .timeline::before {
            content: '';
            position: absolute;
            top: 6px;
            left: 11px;
            width: 2px;
            height: calc(100% - 24px);
            background: rgba(255, 255, 255, 0.08);
        }

        .timeline-step {
            position: relative;
            margin-bottom: 30px;
        }

        .timeline-node {
            position: absolute;
            left: -32px;
            top: 2px;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #1e293b;
            border: 2px solid #475569;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }

        .timeline-node.success {
            background: var(--success);
            border-color: var(--success);
            box-shadow: 0 0 12px var(--success-glow);
        }

        .timeline-node.active {
            background: var(--primary);
            border-color: var(--primary);
            box-shadow: 0 0 12px var(--primary-glow);
        }

        .timeline-node.warning {
            background: var(--warning);
            border-color: var(--warning);
            box-shadow: 0 0 12px var(--warning-glow);
        }

        .timeline-node svg {
            width: 12px;
            height: 12px;
            fill: white;
        }

        .timeline-content {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .timeline-title {
            font-size: 15px;
            font-weight: 700;
            color: #ffffff;
        }

        .timeline-desc {
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.5;
        }

        /* Code/JSON View in Drawer */
        .drawer-section-title {
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 12px;
        }

        .json-pre {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--panel-border);
            padding: 18px;
            border-radius: 16px;
            font-family: monospace;
            font-size: 12px;
            color: #a7f3d0;
            overflow-x: auto;
            max-height: 250px;
        }

        /* Empty State */
        .empty-state {
            grid-column: 1 / -1;
            text-align: center;
            padding: 80px 20px;
            background: var(--panel);
            border: 1px dashed var(--panel-border);
            border-radius: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }

        .empty-state svg {
            width: 64px;
            height: 64px;
            stroke: var(--text-muted);
        }

        .empty-title {
            font-size: 22px;
            font-weight: 700;
        }

        .empty-desc {
            color: var(--text-muted);
            font-size: 14px;
            max-width: 400px;
            line-height: 1.5;
        }

        /* Loading Spinner */
        .spinner {
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .btn.loading .spinner {
            display: inline-block;
        }
        .btn.loading .btn-text {
            display: none;
        }

        /* Skeleton Loader */
        .skeleton-card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            height: 380px;
            position: relative;
            overflow: hidden;
        }

        .skeleton-card::after {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.03), transparent);
            animation: skeleton-wave 1.6s infinite;
        }

        @keyframes skeleton-wave {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
    </style>
</head>
<body>
    <div class="ambient-glow glow-1"></div>
    <div class="ambient-glow glow-2"></div>

    <header>
        <div class="logo-container">
            <div class="logo-icon">
                <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
            </div>
            <div class="logo-text">Aether Audit</div>
        </div>
        <div class="status-badge">
            <div class="status-dot"></div>
            <span>System Active</span>
        </div>
    </header>

    <main>
        <div class="dashboard-header">
            <h1 class="dashboard-title">Pending Approvals</h1>
            <p class="dashboard-subtitle">Real-time financial audits requiring manager intervention</p>
        </div>

        <div id="cards-container" class="cards-grid">
            <!-- Skeleton cards shown during initial fetch -->
            <div class="skeleton-card"></div>
            <div class="skeleton-card"></div>
            <div class="skeleton-card"></div>
        </div>
    </main>

    <!-- Audit Review Slide-out Drawer -->
    <div id="drawer-overlay" class="drawer-overlay" onclick="closeDrawer()"></div>
    <div id="drawer" class="drawer">
        <div class="drawer-header">
            <h2 class="drawer-title">Compliance Audit</h2>
            <button class="close-btn" onclick="closeDrawer()">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>
        </div>
        <div class="timeline" id="timeline-container">
            <!-- Dynamically populated timeline -->
        </div>
        <div>
            <div class="drawer-section-title">Raw Session State</div>
            <pre class="json-pre" id="json-view"></pre>
        </div>
    </div>

    <script>
        let pendingData = [];

        // Fetch pending approvals on load
        async function fetchPending() {
            const container = document.getElementById('cards-container');
            try {
                const response = await fetch('/api/pending');
                if (!response.ok) throw new Error('API fetch failed');
                pendingData = response.ok ? await response.json() : [];
                renderCards();
            } catch (err) {
                console.error(err);
                container.innerHTML = `
                    <div class="empty-state">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                        <div class="empty-title">Error Loading Approvals</div>
                        <div class="empty-desc">Could not connect to the backend service. Check the environment configuration and terminal logs.</div>
                    </div>
                `;
            }
        }

        // Render cards
        function renderCards() {
            const container = document.getElementById('cards-container');
            if (pendingData.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
                        <div class="empty-title">All Clear</div>
                        <div class="empty-desc">No expense reports require human approval at this moment. You are completely caught up!</div>
                    </div>
                `;
                return;
            }

            container.innerHTML = pendingData.map((item, idx) => {
                const expense = item.expense || {};
                const risk = item.risk_assessment || {};
                const isThreat = item.security_event;
                
                // Construct category class
                const category = (expense.category || 'General').toLowerCase();
                let catClass = 'badge-general';
                if (category.includes('meal') || category.includes('food')) catClass = 'badge-meals';
                else if (category.includes('travel') || category.includes('hotel') || category.includes('flight')) catClass = 'badge-travel';
                else if (category.includes('software') || category.includes('saas') || category.includes('cloud')) catClass = 'badge-software';

                // Format amount
                const formattedAmt = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(expense.amount || 0);

                // Build audit panel details
                let auditPanel = '';
                if (isThreat) {
                    auditPanel = `
                        <div class="audit-summary-box audit-threat">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01"/></svg>
                            <div>
                                <strong style="display:block; margin-bottom: 2px;">PROMPT INJECTION DETECTED</strong>
                                Description has keyword anomalies. Direct bypass of automated LLM checks to manager dashboard.
                            </div>
                        </div>
                    `;
                } else if (risk.has_risk) {
                    auditPanel = `
                        <div class="audit-summary-box audit-risk-high">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                            <div>
                                <strong style="display:block; margin-bottom: 2px;">AUDIT RISK IDENTIFIED</strong>
                                ${risk.assessment_summary || 'Anomalies flagged during audit.'}
                            </div>
                        </div>
                    `;
                } else {
                    auditPanel = `
                        <div class="audit-summary-box audit-risk-clean">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                            <div>
                                <strong style="display:block; margin-bottom: 2px;">COMPLIANCE VERIFIED</strong>
                                ${risk.assessment_summary || 'Automated LLM audit reported zero anomalies.'}
                            </div>
                        </div>
                    `;
                }

                // Submitter Initial
                const initial = (expense.submitter || 'U').charAt(0).toUpperCase();

                return `
                    <div class="card ${isThreat ? 'security-threat' : ''}" data-idx="${idx}">
                        <div class="card-header">
                            <span class="category-badge ${catClass}">${expense.category || 'General'}</span>
                            <span class="card-date">${expense.date || ''}</span>
                        </div>
                        <div class="submitter-row">
                            <div class="avatar">${initial}</div>
                            <div class="submitter-email">${expense.submitter || 'Unknown Submitter'}</div>
                        </div>
                        <div class="expense-desc">"${expense.description || 'No description provided'}"</div>
                        <div class="expense-amount">${formattedAmt}</div>
                        
                        ${auditPanel}

                        <div class="card-actions">
                            <button class="btn btn-audit" onclick="openAudit(${idx})">View Review</button>
                            <button class="btn btn-reject" onclick="handleAction(${idx}, false, this)">
                                <span class="btn-text">Reject</span>
                                <div class="spinner"></div>
                            </button>
                            <button class="btn btn-approve" onclick="handleAction(${idx}, true, this)">
                                <span class="btn-text">Approve</span>
                                <div class="spinner"></div>
                            </button>
                        </div>
                    </div>
                `;
            }).join('');

            // Re-apply hover effect listener
            document.querySelectorAll('.card').forEach(card => {
                card.addEventListener('mousemove', e => {
                    const rect = card.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    card.style.setProperty('--x', `${x}px`);
                    card.style.setProperty('--y', `${y}px`);
                });
            });
        }

        // Action handler (Approve / Reject)
        async function handleAction(idx, approved, buttonEl) {
            const cardData = pendingData[idx];
            const cardEl = buttonEl.closest('.card');
            
            const passcode = prompt("Please enter the Manager Passcode to authorize this action:");
            if (passcode === null || passcode.trim() === "") {
                return; // User cancelled or left empty
            }

            // Put card into loading state
            const buttons = cardEl.querySelectorAll('.btn');
            buttons.forEach(btn => btn.disabled = true);
            buttonEl.classList.add('loading');

            try {
                const response = await fetch(`/api/action/${cardData.session_id}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        approved: approved,
                        interrupt_id: cardData.interrupt_id,
                        passcode: passcode
                    })
                });

                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    throw new Error(errData.detail || 'Action request failed');
                }
                
                // Animation of sliding out the card on success
                cardEl.style.transition = 'all 0.6s cubic-bezier(0.16, 1, 0.3, 1)';
                cardEl.style.opacity = '0';
                cardEl.style.transform = 'translateY(30px) scale(0.95)';
                
                setTimeout(() => {
                    pendingData.splice(idx, 1);
                    renderCards();
                }, 600);

            } catch (err) {
                console.error(err);
                alert('Action rejected: ' + err.message);
                buttons.forEach(btn => btn.disabled = false);
                buttonEl.classList.remove('loading');
            }
        }

        // Open Audit Drawer
        function openAudit(idx) {
            const data = pendingData[idx];
            const timeline = document.getElementById('timeline-container');
            const jsonView = document.getElementById('json-view');
            
            // Format state
            const stateObj = {
                session_id: data.session_id,
                user_id: data.user_id,
                expense: data.expense,
                risk_assessment: data.risk_assessment,
                security_event: data.security_event,
                redacted_categories: data.redacted_categories
            };
            jsonView.textContent = JSON.stringify(stateObj, null, 2);

            // Populate Timeline Nodes
            let timelineHtml = `
                <div class="timeline-step">
                    <div class="timeline-node success">
                        <svg viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
                    </div>
                    <div class="timeline-content">
                        <div class="timeline-title">Node 1: Parse Expense</div>
                        <div class="timeline-desc">Extracted amount $${data.expense?.amount} submitted by ${data.expense?.submitter}. Category routed to review.</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-node ${data.security_event ? 'warning' : 'success'}">
                        <svg viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                    </div>
                    <div class="timeline-content">
                        <div class="timeline-title">Node 2: Security Gate</div>
                        <div class="timeline-desc">
                            ${data.security_event ? '⚠️ <strong>Prompt injection pattern matching flagged!</strong> Automated LLM audits bypassed.' : 'Gateway checks passed. Zero injection triggers matching.'}
                            ${data.redacted_categories.length > 0 ? `<br>🔒 PII Redactions active: ${data.redacted_categories.join(', ')}` : ''}
                        </div>
                    </div>
                </div>
            `;

            if (!data.security_event) {
                const risk = data.risk_assessment || {};
                timelineHtml += `
                    <div class="timeline-step">
                        <div class="timeline-node ${risk.has_risk ? 'warning' : 'success'}">
                            <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
                        </div>
                        <div class="timeline-content">
                            <div class="timeline-title">Node 3: LLM Risk Assessment</div>
                            <div class="timeline-desc">
                                <strong>Risk Flagged:</strong> ${risk.has_risk ? 'Yes' : 'No'}<br>
                                ${risk.assessment_summary || 'No risk flags matched.'}
                            </div>
                        </div>
                    </div>
                `;
            }

            timelineHtml += `
                <div class="timeline-step">
                    <div class="timeline-node active">
                        <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
                    </div>
                    <div class="timeline-content">
                        <div class="timeline-title">Node 4: Manager Verdict</div>
                        <div class="timeline-desc">Currently paused. Awaiting manual approve/reject override decision.</div>
                    </div>
                </div>
            `;

            timeline.innerHTML = timelineHtml;

            // Open Drawer
            document.getElementById('drawer').classList.add('open');
            document.getElementById('drawer-overlay').classList.add('open');
        }

        // Close Drawer
        function closeDrawer() {
            document.getElementById('drawer').classList.remove('open');
            document.getElementById('drawer-overlay').classList.remove('open');
        }

        // Initial load
        fetchPending();
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
