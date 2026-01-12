"""
Planner Agent - Optimized for 100% scheduling with dedicated lab rooms.
Implements consecutive lab slots, smarter slot allocation, AND student conflict checking.
"""
import time
import json
from pathlib import Path
from typing import Optional
from collections import defaultdict

from .base_agent import BaseAgent
from models.data_models import (
    Course, Teacher, SchedulingConfig, TimeSlot, Room,
    ScheduleEntry, TimetableProposal, SessionType, Constraint, Day
)
from utils.student_conflicts import StudentConflictMatrix


class PlannerAgent(BaseAgent):
    """
    Optimized Planner Agent with:
    - Dedicated lab rooms (R22-R28)
    - Consecutive 2-hour lab slots
    - LLM-guided strategy with Tool Integrated Reasoning
    - Student conflict checking (no double-booking students)
    """
    
    def __init__(self, api_key: Optional[str] = None, data_dir: str = "."):
        super().__init__(api_key)
        self.lab_room_prefix = "LAB"
        self.regular_room_prefix = "R"
        self.student_conflicts = StudentConflictMatrix(data_dir)
    
    def generate_proposal(
        self,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig,
        constraints: list[Constraint],
        algorithm: str = "llm_driven",
        previous_feedback: Optional[str] = None
    ) -> TimetableProposal:
        """Generate optimized timetable with dedicated labs and consecutive slots."""
        start_time = time.time()
        
        self.log(f"Starting optimized scheduling for {len(courses)} courses...")
        
        # RESET student conflict matrix for fresh scheduling
        self.student_conflicts.reset_schedule()
        
        # Step 1: Get LLM strategy
        strategy = self._get_llm_strategy(courses, config, previous_feedback)
        self.log(f"LLM Strategy: {strategy.get('approach', 'priority-based')}")
        
        # Step 2: Create room pools (21 regular + 7 labs)
        regular_rooms = [Room(room_id=f"R{i}", capacity=config.room_capacity) 
                        for i in range(1, 22)]  # R1-R21
        lab_rooms = [Room(room_id=f"LAB{i}", capacity=config.room_capacity) 
                    for i in range(1, 8)]  # LAB1-LAB7
        
        # Step 3: Schedule with dedicated pools
        entries = self._schedule_optimized(
            courses, teachers, config, strategy, regular_rooms, lab_rooms
        )
        
        # Step 4: Find and fix incomplete courses by reassigning teachers
        unscheduled = self._find_unscheduled(entries, courses)
        if unscheduled:
            self.log(f"Found {len(unscheduled)} incomplete course-batches, attempting teacher reassignment...")
            entries = self._fix_with_teacher_reassignment(
                entries, unscheduled, courses, teachers, config, regular_rooms, lab_rooms
            )
        
        # Step 5: TIR - Use LLM to analyze remaining gaps
        if previous_feedback:
            remaining = self._find_unscheduled(entries, courses)
            if remaining:
                entries = self._tir_fix_gaps(entries, remaining, config, regular_rooms, lab_rooms)
        
        proposal = TimetableProposal(
            proposal_id=f"optimized_{int(time.time())}",
            entries=entries,
            algorithm_used="optimized_llm",
            generation_time_ms=(time.time() - start_time) * 1000
        )
        
        self.log(f"Generated {len(entries)} entries in {proposal.generation_time_ms:.0f}ms")
        return proposal
    
    def _get_llm_strategy(
        self,
        courses: dict[str, Course],
        config: SchedulingConfig,
        previous_feedback: Optional[str]
    ) -> dict:
        """Get LLM strategy guidance."""
        
        total_theory = sum(c.theory_hours * len(c.batches) for c in courses.values())
        total_lab = sum(c.lab_hours * len(c.batches) for c in courses.values())
        
        prompt = f"""You are a scheduling expert. Provide strategy for this problem:

PROBLEM:
- {len(courses)} courses
- {total_theory} theory sessions, {total_lab} lab sessions
- 21 regular rooms, 7 dedicated lab rooms
- 5 days x 8 hours = 40 slots per room
- Labs must be 2 consecutive hours

{f"ISSUES TO FIX: {previous_feedback[:300]}" if previous_feedback else ""}

Respond with JSON only:
{{"approach": "theory_first" or "labs_first" or "balanced",
"spread_across_days": true or false,
"morning_theory": true or false,
"reasoning": "brief"}}"""

        response = self._call_llm(prompt, temperature=0.3)
        
        try:
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except:
            pass
        
        return {"approach": "balanced", "spread_across_days": True, "morning_theory": True}
    
    def _schedule_optimized(
        self,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig,
        strategy: dict,
        regular_rooms: list[Room],
        lab_rooms: list[Room]
    ) -> list[ScheduleEntry]:
        """Optimized scheduling with dedicated lab rooms."""
        entries = []
        
        # Track occupancy - use set for teachers per slot
        teacher_schedule: dict[tuple, set] = defaultdict(set)  # (day, hour) -> set of teacher names
        regular_room_schedule: dict[tuple, bool] = defaultdict(bool)
        lab_room_schedule: dict[tuple, bool] = defaultdict(bool)
        
        # Generate all time slots
        all_slots = config.generate_all_time_slots()
        
        # Sort courses by total hours (larger first - they're hardest to fit)
        sorted_courses = sorted(
            courses.values(),
            key=lambda c: -(c.total_hours * len(c.batches))
        )
        
        # Preference settings
        morning_theory = strategy.get("morning_theory", True)
        spread_days = strategy.get("spread_across_days", True)
        
        for course in sorted_courses:
            for batch_idx, batch_id in enumerate(course.batches):
                teacher_name = course.teacher_assignments.get(batch_id, "TBA")
                student_count = course.batch_sizes[batch_idx] if batch_idx < len(course.batch_sizes) else 60
                
                # --- SCHEDULE LABS FIRST (more constrained - need consecutive 2hr) ---
                if course.lab_hours > 0:
                    lab_slots = sorted(all_slots, key=lambda s: (
                        list(Day).index(s.day),
                        s.hour  # Start from morning to maximize consecutive options
                    ))
                    
                    lab_scheduled = 0
                    i = 0
                    while i < len(lab_slots) - 1 and lab_scheduled < course.lab_hours:
                        slot1 = lab_slots[i]
                        
                        # Find consecutive slot on same day
                        slot2 = None
                        for j in range(i + 1, len(lab_slots)):
                            if (lab_slots[j].day == slot1.day and 
                                lab_slots[j].hour == slot1.hour + 1):
                                slot2 = lab_slots[j]
                                break
                        
                        if not slot2:
                            i += 1
                            continue
                        
                        # Check teacher availability for both hours
                        slot1_key = (slot1.day.value, slot1.hour)
                        slot2_key = (slot2.day.value, slot2.hour)
                        
                        if teacher_name in teacher_schedule[slot1_key] or teacher_name in teacher_schedule[slot2_key]:
                            i += 1
                            continue
                        
                        # Check STUDENT CONFLICTS for both hours
                        if (not self.student_conflicts.check_slot_available(course.code, batch_id, slot1.day.value, slot1.hour) or
                            not self.student_conflicts.check_slot_available(course.code, batch_id, slot2.day.value, slot2.hour)):
                            i += 1
                            continue
                        
                        # Find available lab room for both hours
                        lab_room = None
                        for lr in lab_rooms:
                            key1 = (slot1.day.value, slot1.hour, lr.room_id)
                            key2 = (slot2.day.value, slot2.hour, lr.room_id)
                            if not lab_room_schedule[key1] and not lab_room_schedule[key2]:
                                lab_room = lr
                                break
                        
                        if not lab_room:
                            i += 1
                            continue
                        
                        # Schedule both lab hours
                        for slot in [slot1, slot2]:
                            entry = ScheduleEntry(
                                course_code=course.code,
                                batch_id=batch_id,
                                teacher_name=teacher_name,
                                room_id=lab_room.room_id,
                                time_slot=slot,
                                session_type=SessionType.LAB,
                                student_count=student_count
                            )
                            entries.append(entry)
                            
                            s_key = (slot.day.value, slot.hour)
                            teacher_schedule[s_key].add(teacher_name)
                            lab_room_schedule[(slot.day.value, slot.hour, lab_room.room_id)] = True
                            # Mark in student conflict matrix
                            self.student_conflicts.mark_scheduled(course.code, batch_id, slot.day.value, slot.hour)
                        
                        lab_scheduled += 2
                        i += 2
                
                # --- THEN SCHEDULE THEORY SESSIONS ---
                theory_slots = sorted(all_slots, key=lambda s: (
                    list(Day).index(s.day),
                    s.hour if morning_theory else -s.hour
                ))
                
                theory_scheduled = 0
                for slot in theory_slots:
                    if theory_scheduled >= course.theory_hours:
                        break
                    
                    slot_key = (slot.day.value, slot.hour)
                    
                    # Check teacher availability
                    if teacher_name in teacher_schedule[slot_key]:
                        continue
                    
                    # Check STUDENT CONFLICTS
                    if not self.student_conflicts.check_slot_available(course.code, batch_id, slot.day.value, slot.hour):
                        continue
                    
                    # Find available regular room
                    room = None
                    for r in regular_rooms:
                        room_key = (slot.day.value, slot.hour, r.room_id)
                        if not regular_room_schedule[room_key]:
                            room = r
                            break
                    
                    if not room:
                        continue
                    
                    # Create entry
                    entry = ScheduleEntry(
                        course_code=course.code,
                        batch_id=batch_id,
                        teacher_name=teacher_name,
                        room_id=room.room_id,
                        time_slot=slot,
                        session_type=SessionType.THEORY,
                        student_count=student_count
                    )
                    entries.append(entry)
                    
                    teacher_schedule[slot_key].add(teacher_name)
                    regular_room_schedule[(slot.day.value, slot.hour, room.room_id)] = True
                    # Mark in student conflict matrix
                    self.student_conflicts.mark_scheduled(course.code, batch_id, slot.day.value, slot.hour)
                    theory_scheduled += 1
        
        return entries
    
    def _find_unscheduled(self, entries: list[ScheduleEntry], courses: dict[str, Course]) -> list[dict]:
        """Find courses that aren't fully scheduled."""
        scheduled = defaultdict(lambda: {"theory": 0, "lab": 0})
        
        for entry in entries:
            key = (entry.course_code, entry.batch_id)
            if entry.session_type == SessionType.THEORY:
                scheduled[key]["theory"] += 1
            else:
                scheduled[key]["lab"] += 1
        
        unscheduled = []
        for code, course in courses.items():
            for batch_id in course.batches:
                key = (code, batch_id)
                theory_missing = course.theory_hours - scheduled[key]["theory"]
                lab_missing = course.lab_hours - scheduled[key]["lab"]
                
                if theory_missing > 0 or lab_missing > 0:
                    unscheduled.append({
                        "course": code,
                        "batch": batch_id,
                        "theory_missing": theory_missing,
                        "lab_missing": lab_missing,
                        "original_teacher": course.teacher_assignments.get(batch_id, "TBA")
                    })
        
        return unscheduled
    
    def _fix_with_teacher_reassignment(
        self,
        entries: list[ScheduleEntry],
        unscheduled: list[dict],
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig,
        regular_rooms: list[Room],
        lab_rooms: list[Room]
    ) -> list[ScheduleEntry]:
        """Reassign teachers for incomplete courses - replace ALL sessions with new teacher."""
        
        # Identify which course-batches need reassignment
        courses_to_reassign = set()
        for item in unscheduled:
            courses_to_reassign.add((item["course"], item["batch"]))
        
        # Remove existing entries for courses that will be reassigned
        filtered_entries = []
        for e in entries:
            if (e.course_code, e.batch_id) not in courses_to_reassign:
                filtered_entries.append(e)
        
        # Build occupancy from remaining entries
        teacher_schedule: dict[tuple, set] = defaultdict(set)
        regular_room_schedule: dict[tuple, bool] = defaultdict(bool)
        lab_room_schedule: dict[tuple, bool] = defaultdict(bool)
        
        for e in filtered_entries:
            slot_key = (e.time_slot.day.value, e.time_slot.hour)
            teacher_schedule[slot_key].add(e.teacher_name)
            if e.room_id.startswith("LAB"):
                lab_room_schedule[(e.time_slot.day.value, e.time_slot.hour, e.room_id)] = True
            else:
                regular_room_schedule[(e.time_slot.day.value, e.time_slot.hour, e.room_id)] = True
        
        # Get teacher workloads
        teacher_hours = defaultdict(int)
        for code, course in courses.items():
            for batch_id, teacher in course.teacher_assignments.items():
                teacher_hours[teacher] += course.total_hours
        
        # Get all available teachers sorted by workload (least busy first)
        available_teachers = sorted(teachers.keys(), key=lambda t: teacher_hours[t])
        
        all_slots = config.generate_all_time_slots()
        new_entries = []
        
        for item in unscheduled:
            code = item["course"]
            batch_id = item["batch"]
            original_teacher = item["original_teacher"]
            
            course = courses[code]
            student_count = course.batch_sizes[0] if course.batch_sizes else 60
            
            # Need to schedule ALL sessions for this course-batch
            theory_needed = course.theory_hours
            lab_needed = course.lab_hours
            
            # Try to find a free teacher who can take ALL sessions
            for alt_teacher in available_teachers:
                if alt_teacher == original_teacher:
                    continue  # Skip the original (they're blocked)
                
                potential_slots = []
                
                # Find slots for theory
                for slot in all_slots:
                    if len(potential_slots) >= theory_needed:
                        break
                    slot_key = (slot.day.value, slot.hour)
                    
                    if alt_teacher in teacher_schedule[slot_key]:
                        continue
                    
                    room = None
                    for r in regular_rooms:
                        if not regular_room_schedule[(slot.day.value, slot.hour, r.room_id)]:
                            room = r
                            break
                    
                    # Check student conflicts
                    if room and self.student_conflicts.check_slot_available(code, batch_id, slot.day.value, slot.hour):
                        potential_slots.append((slot, room, SessionType.THEORY))
                
                if len(potential_slots) < theory_needed:
                    continue  # This teacher can't fit all sessions
                
                # Success! Assign this teacher for ALL sessions
                self.log(f"Reassigning {code}-{batch_id}: {original_teacher} -> {alt_teacher} ({theory_needed} sessions)")
                
                for slot, room, session_type in potential_slots[:theory_needed]:
                    entry = ScheduleEntry(
                        course_code=code,
                        batch_id=batch_id,
                        teacher_name=alt_teacher,
                        room_id=room.room_id,
                        time_slot=slot,
                        session_type=session_type,
                        student_count=student_count
                    )
                    new_entries.append(entry)
                    
                    slot_key = (slot.day.value, slot.hour)
                    teacher_schedule[slot_key].add(alt_teacher)
                    regular_room_schedule[(slot.day.value, slot.hour, room.room_id)] = True
                    # Mark in student conflict matrix
                    self.student_conflicts.mark_scheduled(code, batch_id, slot.day.value, slot.hour)
                
                break
        
        return filtered_entries + new_entries
    
    def _tir_fix_gaps(
        self,
        entries: list[ScheduleEntry],
        unscheduled: list[dict],
        config: SchedulingConfig,
        regular_rooms: list[Room],
        lab_rooms: list[Room]
    ) -> list[ScheduleEntry]:
        """Tool Integrated Reasoning - LLM asks for available slots and fills gaps."""
        
        # Build occupancy maps
        teacher_schedule = set()
        room_schedule = set()
        
        for e in entries:
            teacher_schedule.add((e.time_slot.day.value, e.time_slot.hour, e.teacher_name))
            room_schedule.add((e.time_slot.day.value, e.time_slot.hour, e.room_id))
        
        # Find empty slots
        all_slots = config.generate_all_time_slots()
        empty_regular = []
        empty_lab = []
        
        for slot in all_slots:
            for room in regular_rooms:
                if (slot.day.value, slot.hour, room.room_id) not in room_schedule:
                    empty_regular.append({"day": slot.day.value, "hour": slot.hour, "room": room.room_id})
            for room in lab_rooms:
                if (slot.day.value, slot.hour, room.room_id) not in room_schedule:
                    empty_lab.append({"day": slot.day.value, "hour": slot.hour, "room": room.room_id})
        
        # TIR: Ask LLM which unscheduled items to prioritize
        prompt = f"""You have scheduling gaps to fill.

UNSCHEDULED ({len(unscheduled)} items):
{json.dumps(unscheduled[:15], indent=2)}

EMPTY REGULAR SLOTS: {len(empty_regular)}
EMPTY LAB SLOTS: {len(empty_lab)}

Which 5 courses should be prioritized? Respond with JSON array of course codes only:
["CO401", "EE403", ...]"""

        response = self._call_llm(prompt, temperature=0.3)
        self.log(f"TIR priority response: {response[:100]}...")
        
        # For now, return entries as-is (TIR logged but not modifying)
        # A full implementation would parse priorities and reschedule
        return entries
    
    def get_scheduling_stats(self, proposal: TimetableProposal, courses: dict[str, Course]) -> dict:
        """Get statistics about a proposal."""
        total_required = sum(c.total_hours * len(c.batches) for c in courses.values())
        scheduled = len(proposal.entries)
        
        course_coverage = defaultdict(lambda: {"required": 0, "scheduled": 0})
        for code, course in courses.items():
            course_coverage[code]["required"] = course.total_hours * len(course.batches)
        
        for entry in proposal.entries:
            course_coverage[entry.course_code]["scheduled"] += 1
        
        fully_scheduled = sum(
            1 for stats in course_coverage.values()
            if stats["scheduled"] >= stats["required"]
        )
        
        return {
            "total_required_sessions": total_required,
            "total_scheduled_sessions": scheduled,
            "coverage_percentage": round(scheduled / total_required * 100, 2) if total_required > 0 else 0,
            "fully_scheduled_courses": fully_scheduled,
            "partially_scheduled_courses": len(courses) - fully_scheduled,
            "algorithm_used": proposal.algorithm_used,
            "generation_time_ms": round(proposal.generation_time_ms, 2)
        }
