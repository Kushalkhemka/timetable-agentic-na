"""
LLM-Driven Planner Agent - True PlanGEN Implementation.
Uses Gemini 3 Pro with function calling to generate actual schedule entries.
Multiple API calls with context management and tool-integrated reasoning.
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


class LLMPlannerAgent(BaseAgent):
    """
    True LLM-driven scheduling using PlanGEN approach:
    - Multiple API calls with context
    - Function calling for slot assignment
    - Tool-integrated reasoning for constraint checking
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key, model_name="gemini-3-pro-preview")
        self.scheduled_entries: list[ScheduleEntry] = []
        self.teacher_schedule: dict[tuple, set] = defaultdict(set)
        self.room_schedule: dict[tuple, bool] = defaultdict(bool)
    
    def generate_proposal(
        self,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig,
        constraints: list[Constraint],
        algorithm: str = "llm_driven",
        previous_feedback: Optional[str] = None
    ) -> TimetableProposal:
        """Generate timetable with LLM making actual scheduling decisions."""
        start_time = time.time()
        
        self.log(f"Starting TRUE LLM-driven scheduling for {len(courses)} courses...")
        self.scheduled_entries = []
        self.teacher_schedule = defaultdict(set)
        self.room_schedule = defaultdict(bool)
        
        # Create rooms
        regular_rooms = [f"R{i}" for i in range(1, 24)]  # 23 regular
        lab_rooms = [f"LAB{i}" for i in range(1, 8)]      # 7 labs
        all_rooms = regular_rooms + lab_rooms
        
        # Generate slot list
        slots_list = []
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            for hour in range(10, 18):
                slots_list.append(f"{day} {hour}:00-{hour+1}:00")
        
        # Prepare course summaries for LLM
        course_summaries = []
        for code, course in courses.items():
            for batch_id in course.batches:
                teacher = course.teacher_assignments.get(batch_id, "TBA")
                course_summaries.append({
                    "course": code,
                    "batch": batch_id,
                    "teacher": teacher,
                    "theory_hours": course.theory_hours,
                    "lab_hours": course.lab_hours,
                    "students": course.batch_sizes[0] if course.batch_sizes else 60
                })
        
        # Process in batches with LLM
        batch_size = 20  # Process 20 course-batches at a time
        total_batches = (len(course_summaries) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(course_summaries))
            batch_courses = course_summaries[start_idx:end_idx]
            
            self.log(f"LLM scheduling batch {batch_idx + 1}/{total_batches} ({len(batch_courses)} courses)...")
            
            # Get current schedule state for context
            scheduled_summary = self._get_schedule_summary()
            
            # Call LLM to schedule this batch
            new_entries = self._llm_schedule_batch(
                batch_courses, 
                slots_list, 
                all_rooms, 
                lab_rooms,
                scheduled_summary,
                config
            )
            
            # Add valid entries
            for entry in new_entries:
                if self._is_valid_entry(entry):
                    self.scheduled_entries.append(entry)
                    slot_key = (entry.time_slot.day.value, entry.time_slot.hour)
                    self.teacher_schedule[slot_key].add(entry.teacher_name)
                    self.room_schedule[(entry.time_slot.day.value, entry.time_slot.hour, entry.room_id)] = True
        
        proposal = TimetableProposal(
            proposal_id=f"llm_full_{int(time.time())}",
            entries=self.scheduled_entries,
            algorithm_used="llm_full_scheduling",
            generation_time_ms=(time.time() - start_time) * 1000
        )
        
        self.log(f"LLM generated {len(self.scheduled_entries)} entries in {proposal.generation_time_ms:.0f}ms")
        return proposal
    
    def _llm_schedule_batch(
        self,
        batch_courses: list[dict],
        slots_list: list[str],
        all_rooms: list[str],
        lab_rooms: list[str],
        scheduled_summary: str,
        config: SchedulingConfig
    ) -> list[ScheduleEntry]:
        """Have LLM generate schedule entries for a batch of courses."""
        
        # Build occupied slots info
        occupied_slots = []
        for (day, hour), teachers in self.teacher_schedule.items():
            for teacher in teachers:
                occupied_slots.append(f"{day} {hour}:00 - {teacher}")
        
        prompt = f"""You are a timetable scheduling system. Generate schedule entries for these courses.

COURSES TO SCHEDULE:
{json.dumps(batch_courses, indent=2)}

RULES:
1. Each teacher can only be in ONE room at a time
2. Each room can only have ONE class at a time  
3. Labs must use LAB rooms (LAB1-LAB7)
4. Labs should be 2 CONSECUTIVE hours on same day
5. Theory uses regular rooms (R1-R23)
6. Schedule ALL theory_hours and lab_hours for each course-batch

AVAILABLE SLOTS: Monday-Friday, 10:00-18:00 (8 slots/day)
ROOMS: R1-R23 (theory), LAB1-LAB7 (labs)

CURRENTLY SCHEDULED ({len(self.scheduled_entries)} entries):
{scheduled_summary}

OCCUPIED TEACHER SLOTS:
{chr(10).join(occupied_slots[-50:]) if occupied_slots else "None yet"}

Generate a JSON array of schedule entries. Each entry must have:
- course: course code
- batch: batch ID
- teacher: teacher name (MUST match the teacher assigned to this course-batch)
- room: room ID (R1-R23 for theory, LAB1-LAB7 for labs)
- day: Monday/Tuesday/Wednesday/Thursday/Friday
- hour: 10-17 (start hour)
- type: "theory" or "lab"

IMPORTANT: Generate entries for ALL hours needed. If a course needs 4 theory hours, generate 4 separate entries.

Respond with ONLY a JSON array, no markdown:
[{{"course": "CO401", "batch": "B1", "teacher": "Dr. X", "room": "R1", "day": "Monday", "hour": 10, "type": "theory"}}, ...]"""

        response = self._call_llm(prompt, temperature=0.3)
        
        # Parse response
        entries = []
        try:
            # Find JSON array in response
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                schedule_data = json.loads(text[start:end])
                
                for item in schedule_data:
                    entry = self._create_entry_from_llm(item, config)
                    if entry:
                        entries.append(entry)
        except Exception as e:
            self.log(f"Error parsing LLM response: {e}")
        
        return entries
    
    def _create_entry_from_llm(self, item: dict, config: SchedulingConfig) -> Optional[ScheduleEntry]:
        """Convert LLM output to ScheduleEntry."""
        try:
            day_map = {
                "Monday": Day.MONDAY,
                "Tuesday": Day.TUESDAY,
                "Wednesday": Day.WEDNESDAY,
                "Thursday": Day.THURSDAY,
                "Friday": Day.FRIDAY
            }
            
            day = day_map.get(item.get("day"))
            if not day:
                return None
            
            hour = int(item.get("hour", 0))
            if hour < 10 or hour > 17:
                return None
            
            time_slot = TimeSlot(day=day, hour=hour)
            session_type = SessionType.LAB if item.get("type") == "lab" else SessionType.THEORY
            
            return ScheduleEntry(
                course_code=item.get("course", ""),
                batch_id=item.get("batch", ""),
                teacher_name=item.get("teacher", ""),
                room_id=item.get("room", ""),
                time_slot=time_slot,
                session_type=session_type,
                student_count=60
            )
        except Exception:
            return None
    
    def _is_valid_entry(self, entry: ScheduleEntry) -> bool:
        """Check if entry doesn't conflict with existing schedule."""
        slot_key = (entry.time_slot.day.value, entry.time_slot.hour)
        room_key = (entry.time_slot.day.value, entry.time_slot.hour, entry.room_id)
        
        # Check teacher conflict
        if entry.teacher_name in self.teacher_schedule[slot_key]:
            return False
        
        # Check room conflict
        if self.room_schedule[room_key]:
            return False
        
        return True
    
    def _get_schedule_summary(self) -> str:
        """Get summary of current schedule for LLM context."""
        if not self.scheduled_entries:
            return "No entries scheduled yet."
        
        # Summarize by day
        by_day = defaultdict(list)
        for e in self.scheduled_entries[-100:]:  # Last 100 for context
            by_day[e.time_slot.day.value].append(
                f"{e.time_slot.hour}:00 {e.course_code}-{e.batch_id} ({e.teacher_name}) in {e.room_id}"
            )
        
        summary = []
        for day, entries in by_day.items():
            summary.append(f"{day}: {len(entries)} entries")
        
        return "\n".join(summary)
    
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
