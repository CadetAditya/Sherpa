"""
CommitCon Driver Copilot - Standalone Terminal CLI & Script Interface
Runs the fine-tuned driver assistant locally via Ollama or the local FastAPI backend.
Modeled on the Supervised project standalone execution mode.
"""
import sys
import os
import json
import httpx
from pathlib import Path

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/chat")

def ask_driver_copilot(prompt: str, zone_id: str = "Z06", driver_id: str = "DRV_101") -> str:
    """Queries the local driver assistant via Ollama or the CommitCon local backend."""
    # 1. Try local Ollama if running
    try:
        ollama_payload = {
            "model": "project-driver:latest",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are Sherpa, an AI driving assistant for the CommitCon platform in Bengaluru. Driver is in Zone {zone_id}."
                },
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        resp = httpx.post(OLLAMA_URL, json=ollama_payload, timeout=10.0)
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "").strip()
    except Exception:
        pass

    # 2. Try CommitCon local backend (/api/chat)
    try:
        backend_payload = {
            "driver_id": driver_id,
            "driver_prompt": prompt,
            "current_zone_id": zone_id,
            "history": []
        }
        resp = httpx.post(BACKEND_URL, json=backend_payload, timeout=10.0)
        if resp.status_code == 200:
            return resp.json().get("answer", "").strip()
    except Exception as exc:
        return f"Error communicating with local driver engine: {exc}"

    return "No local assistant service responded. Ensure the local server is running (python run.py)."

def interactive_session():
    print("=" * 65)
    print("🧭 CommitCon Sherpa Driver Copilot - Standalone CLI")
    print("=" * 65)
    print("Type your questions as a driver (e.g. 'Where should I go?', 'Exit' to quit)\n")

    zone = "Z06"
    while True:
        try:
            user_input = input(f"Driver [{zone}] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting Sherpa CLI. Drive safely!")
                break
            
            print("Thinking...")
            answer = ask_driver_copilot(user_input, zone_id=zone)
            print(f"\nSherpa 🧭:\n{answer}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Sherpa CLI.")
            break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(ask_driver_copilot(question))
    else:
        interactive_session()
