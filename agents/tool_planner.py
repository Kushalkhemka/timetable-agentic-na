"""
LLM Planner with Function Calling - True PlanGEN Implementation.
Uses Gemini 3 Pro with tool/function calling for real-time constraint checking.
"""
import time
import json
from typing import Optional
from collections import defaultdict

from .base_agent import BaseAgent
from models.data_models import (
    Course, Teacher, SchedulingConfig, TimeSlot, Room,
    ScheduleEntry, TimetableProposal, SessionType, Constraint, Day
)
from google.genai import types


class ToolBasedPlannerAgent(BaseAgent):
    """
    LLM-driven scheduling with function calling:
    - LLM can call check_slot_available() to verify constraints
    - LLM can call assign_slot() to make assignments
    - Real-time constraint checking prevents conflicts
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key, model_name="gemini-3-pro-preview")
        self.reset_state()
    
    def reset_state(self):
        """Reset scheduling state."""
        self.scheduled_entries: list[ScheduleEntry] = []
        self.teacher_schedule: dict[tuple, str] = {}  # (day, hour) -> teacher
        self.room_schedule: dict[tuple, bool] = defaultdict(bool)  # (day, hour, room) -> occupied
        self.course_progress: dict[str, dict] = {}  # course-batch -> {theory: 0, lab: 0}
    
    def generate_proposal(
        self,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig,
        constraints: list[Constraint],
        algorithm: str = "tool_based",
        previous_feedback: Optional[str] = None
    ) -> TimetableProposal:
        """Generate timetable using LLM with function calling."""
        start_time = time.time()
        
        self.log(f"Starting TOOL-BASED LLM scheduling for {len(courses)} courses...")
        self.reset_state()
        
        # Prepare course list
        course_list = []
        for code, course in courses.items():
            for batch_id in course.batches:
                teacher = course.teacher_assignments.get(batch_id, "TBA")
                course_list.append({
                    "id": f"{code}-{batch_id}",
                    "course": code,
                    "batch": batch_id,
                    "teacher": teacher,
                    "theory_hours": course.theory_hours,
                    "lab_hours": course.lab_hours
                })
                self.course_progress[f"{code}-{batch_id}"] = {"theory": 0, "lab": 0}
        
        # Define tools for the LLM
        tools = self._define_tools()
        
        # Process in conversation with tools
        self.log(f"Processing {len(course_list)} course-batches with tool calling...")
        
        # Build initial prompt with all courses and rules
        system_prompt = self._build_system_prompt(course_list, config)
        
        # Multi-turn scheduling with tool use
        self._schedule_with_tools(system_prompt, course_list, config, tools)
        
        proposal = TimetableProposal(
            proposal_id=f"tool_based_{int(time.time())}",
            entries=self.scheduled_entries,
            algorithm_used="tool_based_llm",
            generation_time_ms=(time.time() - start_time) * 1000
        )
        
        self.log(f"Generated {len(self.scheduled_entries)} entries in {proposal.generation_time_ms:.0f}ms")
        return proposal
    
    def _define_tools(self) -> list:
        """Define function calling tools for the LLM."""
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="check_slot_available",
                        description="Check if a time slot is available for a teacher and room",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "day": types.Schema(type="STRING", description="Day: Monday/Tuesday/Wednesday/Thursday/Friday"),
                                "hour": types.Schema(type="INTEGER", description="Hour: 10-17"),
                                "teacher": types.Schema(type="STRING", description="Teacher name"),
                                "room": types.Schema(type="STRING", description="Room ID: R1-R23 or LAB1-LAB7")
                            },
                            required=["day", "hour", "teacher", "room"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="assign_slot",
                        description="Assign a course-batch to a specific time slot",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "course": types.Schema(type="STRING", description="Course code"),
                                "batch": types.Schema(type="STRING", description="Batch ID"),
                                "teacher": types.Schema(type="STRING", description="Teacher name"),
                                "room": types.Schema(type="STRING", description="Room ID"),
                                "day": types.Schema(type="STRING", description="Day of week"),
                                "hour": types.Schema(type="INTEGER", description="Start hour (10-17)"),
                                "session_type": types.Schema(type="STRING", description="theory or lab")
                            },
                            required=["course", "batch", "teacher", "room", "day", "hour", "session_type"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="get_schedule_status",
                        description="Get current scheduling progress and available slots",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={}
                        )
                    )
                ]
            )
        ]
    
    def _build_system_prompt(self, course_list: list, config: SchedulingConfig) -> str:
        """Build the initial system prompt."""
        return f"""You are a timetable scheduling AI. Schedule ALL courses using the tools provided.

COURSES TO SCHEDULE ({len(course_list)} total):
{json.dumps(course_list[:30], indent=2)}
{"..." if len(course_list) > 30 else ""}

RULES:
1. Each teacher can only be in ONE room at a time
2. Each room can only have ONE class at a time
3. Labs MUST use LAB1-LAB7, Theory MUST use R1-R23
4. Labs should be scheduled as 2 CONSECUTIVE hours on the same day
5. Schedule ALL theory_hours and lab_hours for each course

AVAILABLE:
- Days: Monday-Friday
- Hours: 10-17 (8 slots per day)  
- Theory Rooms: R1-R23 (23 rooms)
- Lab Rooms: LAB1-LAB7 (7 rooms)

WORKFLOW:
1. For each course-batch, use check_slot_available() to find free slots
2. Use assign_slot() to make assignments
3. Use get_schedule_status() periodically to check progress

Start scheduling now. Begin with courses that have labs (they're more constrained)."""
    
    def _schedule_with_tools(self, system_prompt: str, course_list: list, config: SchedulingConfig, tools: list):
        """Run multi-turn scheduling with tool calling."""
        
        contents = [types.Content(role="user", parts=[types.Part(text=system_prompt)])]
        
        generate_config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=65535,
            tools=tools,
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        )
        
        max_turns = 50  # Limit turns to prevent infinite loops
        total_assigned = 0
        
        for turn in range(max_turns):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=generate_config,
                )
                
                # Log this API call
                response_text = ""
                function_calls = []
                
                if response.candidates and response.candidates[0].content:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call)
                
                # Log the call
                BaseAgent._logger.log_call(
                    agent_name=self.agent_name,
                    prompt=f"Turn {turn + 1}: {len(function_calls)} function calls",
                    response=response_text[:500] if response_text else f"Function calls: {[fc.name for fc in function_calls]}",
                    thinking=None,
                    temperature=0.3,
                    duration_ms=0,
                    success=True
                )
                
                if not function_calls:
                    # LLM finished or gave text response
                    if "complete" in response_text.lower() or total_assigned >= len(course_list) * 4:
                        self.log(f"Scheduling complete after {turn + 1} turns")
                        break
                    # Add response and continue
                    contents.append(types.Content(role="model", parts=[types.Part(text=response_text)]))
                    contents.append(types.Content(role="user", parts=[types.Part(text="Continue scheduling. Use the tools.")]))
                    continue
                
                # Process function calls
                function_responses = []
                for fc in function_calls:
                    result = self._execute_function(fc.name, fc.args)
                    if fc.name == "assign_slot" and result.get("success"):
                        total_assigned += 1
                    function_responses.append(
                        types.Part(function_response=types.FunctionResponse(
                            name=fc.name,
                            response=result
                        ))
                    )
                
                # Add model's function calls and our responses
                contents.append(response.candidates[0].content)
                contents.append(types.Content(role="user", parts=function_responses))
                
                if turn % 10 == 0:
                    self.log(f"Turn {turn + 1}: {total_assigned} slots assigned, {len(self.scheduled_entries)} entries")
                    
            except Exception as e:
                self.log(f"Error in turn {turn + 1}: {e}")
                break
        
        self.log(f"Completed {turn + 1} turns, {total_assigned} assignments made")
    
    def _execute_function(self, name: str, args: dict) -> dict:
        """Execute a function called by the LLM."""
        if name == "check_slot_available":
            return self._check_slot(args)
        elif name == "assign_slot":
            return self._assign_slot(args)
        elif name == "get_schedule_status":
            return self._get_status()
        return {"error": f"Unknown function: {name}"}
    
    def _check_slot(self, args: dict) -> dict:
        """Check if a slot is available."""
        day = args.get("day", "")
        hour = int(args.get("hour", 0))
        teacher = args.get("teacher", "")
        room = args.get("room", "")
        
        # Check teacher availability
        teacher_key = (day, hour)
        if teacher_key in self.teacher_schedule:
            if self.teacher_schedule[teacher_key] == teacher:
                return {"available": False, "reason": f"{teacher} already scheduled at {day} {hour}:00"}
        
        # Check room availability
        room_key = (day, hour, room)
        if self.room_schedule[room_key]:
            return {"available": False, "reason": f"Room {room} already occupied at {day} {hour}:00"}
        
        return {"available": True, "message": f"Slot {day} {hour}:00 in {room} is available for {teacher}"}
    
    def _assign_slot(self, args: dict) -> dict:
        """Assign a course to a slot."""
        course = args.get("course", "")
        batch = args.get("batch", "")
        teacher = args.get("teacher", "")
        room = args.get("room", "")
        day = args.get("day", "")
        hour = int(args.get("hour", 0))
        session_type = args.get("session_type", "theory")
        
        # Validate
        if hour < 10 or hour > 17:
            return {"success": False, "error": "Hour must be 10-17"}
        
        # Check availability first
        check_result = self._check_slot({"day": day, "hour": hour, "teacher": teacher, "room": room})
        if not check_result.get("available", False):
            return {"success": False, "error": check_result.get("reason", "Slot not available")}
        
        # Create entry
        day_map = {"Monday": Day.MONDAY, "Tuesday": Day.TUESDAY, "Wednesday": Day.WEDNESDAY,
                   "Thursday": Day.THURSDAY, "Friday": Day.FRIDAY}
        
        if day not in day_map:
            return {"success": False, "error": f"Invalid day: {day}"}
        
        entry = ScheduleEntry(
            course_code=course,
            batch_id=batch,
            teacher_name=teacher,
            room_id=room,
            time_slot=TimeSlot(day=day_map[day], hour=hour),
            session_type=SessionType.LAB if session_type == "lab" else SessionType.THEORY,
            student_count=60
        )
        
        # Mark occupied
        self.teacher_schedule[(day, hour)] = teacher
        self.room_schedule[(day, hour, room)] = True
        self.scheduled_entries.append(entry)
        
        # Update progress
        key = f"{course}-{batch}"
        if key in self.course_progress:
            if session_type == "lab":
                self.course_progress[key]["lab"] += 1
            else:
                self.course_progress[key]["theory"] += 1
        
        return {
            "success": True, 
            "message": f"Assigned {course}-{batch} to {day} {hour}:00 in {room}",
            "total_entries": len(self.scheduled_entries)
        }
    
    def _get_status(self) -> dict:
        """Get current scheduling status."""
        incomplete = []
        for key, progress in self.course_progress.items():
            # This is simplified - in real implementation we'd track required vs scheduled
            if progress["theory"] < 4 and progress["lab"] == 0:
                incomplete.append(key)
        
        return {
            "total_scheduled": len(self.scheduled_entries),
            "incomplete_courses": len(incomplete),
            "sample_incomplete": incomplete[:5]
        }
    
    def get_scheduling_stats(self, proposal: TimetableProposal, courses: dict[str, Course]) -> dict:
        """Get statistics about a proposal."""
        total_required = sum(c.total_hours * len(c.batches) for c in courses.values())
        scheduled = len(proposal.entries)
        
        return {
            "total_required_sessions": total_required,
            "total_scheduled_sessions": scheduled,
            "coverage_percentage": round(scheduled / total_required * 100, 2) if total_required > 0 else 0,
            "algorithm_used": proposal.algorithm_used,
            "generation_time_ms": round(proposal.generation_time_ms, 2)
        }
