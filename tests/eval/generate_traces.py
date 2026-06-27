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
import json
import asyncio
import sys
from contextlib import aclosing
from google.adk.runners import InMemoryRunner
from google.genai import types
from fastapi.encoders import jsonable_encoder
from expense_agent.agent import app as agent_app

# Load .env file if present
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
if os.path.exists(dotenv_path):
    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

async def generate():
    dataset_path = "tests/eval/datasets/basic-dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        sys.exit(1)
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    runner = InMemoryRunner(app=agent_app)
    output_cases = []
    
    for case in data["eval_cases"]:
        case_id = case["eval_case_id"]
        prompt_text = case["prompt"]["parts"][0]["text"]
        print(f"Generating trace for case: {case_id}")
        
        session_id = f"eval-{case_id}"
        
        # Create session on the runner
        await runner.session_service.create_session(
            app_name=agent_app.name,
            user_id="eval-user",
            session_id=session_id
        )
        
        events_list = [
            {
                "author": "user",
                "content": {
                    "role": "user",
                    "parts": [{"text": prompt_text}]
                }
            }
        ]
        
        new_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text)]
        )
        
        paused = False
        async with aclosing(
            runner.run_async(
                user_id="eval-user",
                session_id=session_id,
                new_message=new_message
            )
        ) as agen:
            async for event in agen:
                if event.content:
                    events_list.append({
                        "author": event.author or "expense_agent",
                        "content": jsonable_encoder(event.content)
                    })
                    # Intercept the human approval pause
                    if event.content.parts:
                        for part in event.content.parts:
                            if part.function_call and part.function_call.name == "adk_request_input":
                                paused = True
        
        if paused:
            # Automate decision: reject injection, approve others
            decision = "REJECTED" if "injection" in case_id else "APPROVED"
            print(f"  Workflow paused. Automating human decision: {decision}")
            
            # Record human event in trace
            human_response_part = {
                "function_response": {
                    "name": "adk_request_input",
                    "response": {"decision": decision}
                }
            }
            events_list.append({
                "author": "user",
                "content": {
                    "role": "user",
                    "parts": [human_response_part]
                }
            })
            
            # Resume run
            resume_message = types.Content(
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
            
            async with aclosing(
                runner.run_async(
                    user_id="eval-user",
                    session_id=session_id,
                    new_message=resume_message
                )
            ) as agen:
                async for event in agen:
                    if event.content:
                        events_list.append({
                            "author": event.author or "expense_agent",
                            "content": jsonable_encoder(event.content)
                        })
        
        # Find the final text response from the model
        final_response = None
        for event in reversed(events_list):
            if event.get("author") != "user" and event.get("content"):
                parts = event["content"].get("parts", [])
                if any(p.get("text") for p in parts):
                    final_response = event["content"]
                    break

        output_cases.append({
            "eval_case_id": case_id,
            "prompt": case["prompt"],
            "responses": [{"response": final_response}] if final_response else [],
            "agent_data": {
                "agents": {
                    "expense_agent": {
                        "agent_id": "expense_agent",
                        "instruction": agent_app.root_agent.description or ""
                    }
                },
                "turns": [
                    {
                        "turn_index": 0,
                        "events": events_list
                    }
                ]
            }
        })
        
    output_dir = "artifacts/traces"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_traces.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"eval_cases": output_cases}, f, indent=2)
    print(f"Successfully wrote generated traces to {output_path}")

if __name__ == "__main__":
    asyncio.run(generate())
