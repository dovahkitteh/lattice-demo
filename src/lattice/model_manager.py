"""
Model Manager for handling local LLM backends and dynamic model switching.

Supports two modes:
- Ollama (default): no local server is spawned here; models are switched by using
  the Ollama HTTP API and environment variables to select the active model.
- text-generation-webui (legacy): can spawn and switch models via CLI.

Designed for Windows but uses cross-platform safe calls where possible.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import logging
import threading, os
from datetime import datetime
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Configuration helpers
# ----------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "hermes": {
        "label": "Hermes (8K)",
        "model_path": "Hermes",
        "context_length": 8192,
        "port": 49545,
        "cli_flags": ""
    },
    "pocketdoc_dans": {
        "label": "PocketDoc Dan (4K)",
        "model_path": "PocketDoc_Dans-PersonalityEngine-V1.3.0-24b-Q4_K_L",
        "context_length": 4096,
        "port": 49545,
        "cli_flags": ""
    }
}

_CONFIG_PATH = Path("config/models.json")


def _load_model_catalog() -> Dict[str, Dict[str, Any]]:
    if _CONFIG_PATH.exists():
        try:
            with _CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info("Loaded model catalog from %s", _CONFIG_PATH)
                return data
        except Exception as e:
            logger.error("Failed to read %s: %s", _CONFIG_PATH, e)
    logger.warning("Using default in-code model catalog. Create %s to customise.", _CONFIG_PATH)
    return _DEFAULT_CONFIG.copy()


# ----------------------------------------------------------------------------
# ModelManager singleton
# ----------------------------------------------------------------------------

class ModelManager:
    """Singleton‐style manager for local LLM backend."""

    _instance: "ModelManager | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialised") and self._initialised:
            return
        self._initialised = True

        self.catalog = _load_model_catalog()
        self.current_model: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        # Protect concurrent switches
        self._lock = asyncio.Lock()
        # Status tracking
        self._phase: str = "idle"  # idle | stopping | starting | running | error
        self._last_error: Optional[str] = None
        # Ensure logs dir
        self._logs_dir = Path("logs")
        self._logs_dir.mkdir(exist_ok=True)
        self._current_log_path: Optional[Path] = None

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """Return catalog with current flag."""
        result = {}
        for key, cfg in self.catalog.items():
            result[key] = {**cfg, "is_active": key == self.current_model}
        return result

    async def switch_model(self, model_key: str, timeout: int = 120) -> Dict[str, Any]:
        """Switch running model. For Ollama, requests the model to be loaded.

        Returns status dict with success flag and message.
        """
        async with self._lock:
            self._phase = "idle"
            if model_key not in self.catalog:
                return {"success": False, "error": f"Unknown model '{model_key}'"}

            if model_key == self.current_model:
                return {"success": True, "message": "Model already active", "active_model": model_key}

            # Detect backend from env
            backend = os.getenv("LLM_BACKEND", "auto").lower()
            api_base = os.getenv("LLM_API", "http://127.0.0.1:11434").rstrip("/")
            is_ollama = (backend == "ollama") or (":11434" in api_base) or api_base.endswith("/api")

            if is_ollama:
                # Ask Ollama to load the model (no spawning here)
                self._phase = "starting"
                cfg = self.catalog[model_key]
                model_name = cfg.get("model_path") or cfg.get("label") or model_key
                # Configure env defaults used by LLM client/streaming
                try:
                    os.environ["OLLAMA_MODEL"] = model_name
                    if cfg.get("context_length"):
                        os.environ["OLLAMA_NUM_CTX"] = str(cfg["context_length"])  # e.g., 8192
                except Exception:
                    pass
                try:
                    import aiohttp
                    chat_url = api_base if api_base.endswith("/api/chat") else api_base + "/api/chat"
                    # Send an empty minimal chat to load model
                    payload = {"model": model_name, "messages": [{"role": "user", "content": "Hello"}], "stream": False}
                    timeout = aiohttp.ClientTimeout(total=30, connect=5)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(chat_url, json=payload) as resp:
                            if resp.status in (200, 400, 422):
                                self.current_model = model_key
                                self._phase = "running"
                                return {"success": True, "active_model": model_key}
                            else:
                                txt = await resp.text()
                                self._phase = "error"
                                self._last_error = f"ollama_load_failed:{resp.status}:{txt[:120]}"
                                return {"success": False, "error": self._last_error}
                except Exception as e:
                    self._phase = "error"
                    self._last_error = str(e)
                    return {"success": False, "error": str(e)}
            
            # Legacy: text-generation-webui process management
            # Stop current process if running
            if self.process and self.process.poll() is None:
                logger.info("Terminating existing text-generation-webui process (PID %s)…", self.process.pid)
                self._phase = "stopping"
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                    logger.info("Process terminated.")
                except subprocess.TimeoutExpired:
                    logger.warning("Terminate timeout; killing process…")
                    self.process.kill()

            # Launch new model (text-generation-webui)
            cfg = self.catalog[model_key]
            cmd = self._build_launch_command(cfg)
            cwd = Path("text-generation-webui").resolve()
            logger.info("Launching model '%s' with command: %s", cfg["label"], " ".join(cmd))
            self._phase = "starting"
            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception as e:
                logger.error("Failed to start text-generation-webui: %s", e)
                self._phase = "error"
                self._last_error = str(e)
                return {"success": False, "error": str(e)}

            # Wait for port to open
            port = cfg.get("port", 5000)
            logger.info("Waiting for API port %s to become available…", port)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if _check_port_open(port):
                    logger.info("Model '%s' is now serving on port %s", cfg["label"], port)
                    if _check_api_ready(port):
                        logger.info("OpenAI API responsive on port %s", port)
                        self.current_model = model_key
                        self._phase = "running"
                        return {"success": True, "active_model": model_key, "pid": self.process.pid}
                    else:
                        logger.warning("Port %s open but API not yet ready, continuing wait", port)
                        # continue loop

            # Timeout
            logger.error("Model '%s' failed to start within %ss", cfg["label"], timeout)
            self._phase = "error"
            self._last_error = "startup_timeout"
            return {"success": False, "error": "startup_timeout"}

    def get_status(self) -> Dict[str, Any]:
        """Return current model, process info and phase."""
        return {
            "active_model": self.current_model,
            "pid": self.process.pid if self.process else None,
            "running": self.process.poll() is None if self.process else False,
            "phase": self._phase,
            "error": self._last_error,
        }

    def get_recent_log_tail(self, lines: int = 200) -> str:
        """Return last `lines` lines of current model log."""
        if not self._current_log_path or not self._current_log_path.exists():
            return "No log available yet."
        try:
            with self._current_log_path.open('r', encoding='utf-8', errors='ignore') as f:
                return ''.join(f.readlines()[-lines:])
        except Exception as e:
            return f"Error reading log: {e}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_launch_command(cfg: Dict[str, Any]):
        base_cmd = [sys.executable, "server.py", "--api", "--listen"]
        # Model path/name
        if cfg.get("model_path"):
            base_cmd += ["--model", cfg["model_path"]]
        # Context length / max_seq_len
        if cfg.get("context_length"):
            base_cmd += ["--max_seq_len", str(cfg["context_length"])]
        # Port
        if cfg.get("port"):
            base_cmd += ["--port", str(cfg["port"])]
        # Extra flags
        extra = cfg.get("cli_flags") or ""
        if extra:
            base_cmd += extra.split()
        return base_cmd

    def _capture_process_output(self, proc: subprocess.Popen, model_key: str):
        """Start background thread to pipe stdout/stderr to rotating log files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self._logs_dir / f"textgen_{model_key}_{timestamp}.log"

        handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
        # create mini logger
        proc_logger = logging.getLogger(f"textgen.{model_key}.{timestamp}")
        proc_logger.setLevel(logging.INFO)
        proc_logger.addHandler(handler)

        def _reader(stream, prefix):
            for line in iter(stream.readline, ''):
                if line:
                    proc_logger.info("%s %s", prefix, line.rstrip())
            stream.close()

        threading.Thread(target=_reader, args=(proc.stdout, "[O]"), daemon=True).start()
        threading.Thread(target=_reader, args=(proc.stderr, "[E]"), daemon=True).start()
        self._current_log_path = log_path


# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

import socket

def _check_port_open(port: int) -> bool:
    """Return True if something is listening on localhost:port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _check_api_ready(port: int) -> bool:
    """Return True if OpenAI /v1/chat/completions responds (even with 422)."""
    import http.client, json
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        payload = json.dumps({"model": "test", "messages": [{"role": "user", "content": "test"}], "max_tokens": 1})
        headers = {"Content-Type": "application/json"}
        conn.request("POST", "/v1/chat/completions", body=payload, headers=headers)
        resp = conn.getresponse()
        conn.close()
        return resp.status in (200, 400, 422)
    except Exception:
        return False


# Public singleton access
model_manager = ModelManager() 