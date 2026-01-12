"""
Base agent class with Gemini 3 Pro API integration with Thinking Mode.
Uses the new google.genai package with HIGH thinking level.
"""
from google import genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from typing import Optional


class LLMLogger:
    """Logs all LLM calls with full prompts and responses."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "llm_calls.json"
        self.calls: list[dict] = []
        self._load_existing()
    
    def _load_existing(self):
        if self.log_file.exists():
            try:
                with open(self.log_file) as f:
                    data = json.load(f)
                    self.calls = data.get("calls", [])
            except:
                self.calls = []
    
    def log_call(
        self,
        agent_name: str,
        prompt: str,
        response: str,
        temperature: float,
        duration_ms: float,
        success: bool,
        thinking: Optional[str] = None,
        error: Optional[str] = None
    ):
        call_record = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "prompt": prompt,
            "thinking": thinking,
            "response": response,
            "temperature": temperature,
            "duration_ms": round(duration_ms, 2),
            "prompt_chars": len(prompt),
            "response_chars": len(response) if response else 0,
            "success": success,
            "error": error
        }
        self.calls.append(call_record)
        self._save()
    
    def _save(self):
        recent_calls = self.calls[-100:]
        with open(self.log_file, "w") as f:
            json.dump({"total_calls": len(self.calls), "calls": recent_calls}, f, indent=2)
    
    def get_summary(self) -> dict:
        if not self.calls:
            return {"total_calls": 0}
        return {
            "total_calls": len(self.calls),
            "successful": sum(1 for c in self.calls if c["success"]),
            "failed": sum(1 for c in self.calls if not c["success"]),
            "avg_duration_ms": sum(c["duration_ms"] for c in self.calls) / len(self.calls)
        }
    
    def clear(self):
        self.calls = []
        if self.log_file.exists():
            self.log_file.unlink()


class BaseAgent:
    """Base class for all PlanGEN agents with Gemini 3 Pro + Thinking Mode."""
    
    _logger: Optional[LLMLogger] = None
    _client: Optional[genai.Client] = None
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3-pro-preview"):
        load_dotenv(Path(__file__).parent.parent / ".env")
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        self.model_name = model_name
        self.agent_name = self.__class__.__name__
        
        # Initialize shared client
        if BaseAgent._client is None:
            BaseAgent._client = genai.Client(api_key=self.api_key)
        
        # Initialize shared logger
        if BaseAgent._logger is None:
            log_dir = Path(__file__).parent.parent / "logs"
            BaseAgent._logger = LLMLogger(log_dir)
    
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Make a call to Gemini 3 Pro with HIGH thinking level."""
        import time
        start_time = time.time()
        
        try:
            # Build content
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)]
                )
            ]
            
            # Config with thinking mode HIGH
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=65535,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ],
                thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            )
            
            response = BaseAgent._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            
            # Extract thinking and response parts
            thinking_text = ""
            response_text = ""
            
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        thinking_text += str(part.text) + "\n"
                    elif hasattr(part, 'text') and part.text:
                        response_text += part.text
            
            if not response_text:
                response_text = response.text
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Log with thinking
            BaseAgent._logger.log_call(
                agent_name=self.agent_name,
                prompt=prompt,
                response=response_text,
                thinking=thinking_text if thinking_text else None,
                temperature=temperature,
                duration_ms=duration_ms,
                success=True
            )
            
            return response_text
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            BaseAgent._logger.log_call(
                agent_name=self.agent_name,
                prompt=prompt,
                response="",
                thinking=None,
                temperature=temperature,
                duration_ms=duration_ms,
                success=False,
                error=error_msg
            )
            
            print(f"[{self.agent_name}] LLM Error: {e}")
            return ""
    
    def _parse_json_response(self, response: str) -> dict:
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except:
                    pass
            return {}
    
    def log(self, message: str) -> None:
        print(f"[{self.agent_name}] {message}")
    
    @classmethod
    def get_llm_stats(cls) -> dict:
        if cls._logger:
            return cls._logger.get_summary()
        return {}
