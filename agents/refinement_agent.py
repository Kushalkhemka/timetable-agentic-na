"""
Refinement Agent - Uses LLM to iteratively improve the schedule.
Post-processing optimization to reduce student conflicts.
"""
import json
from typing import Optional
from collections import defaultdict

from .base_agent import BaseAgent
from models.data_models import (
    TimetableProposal, ScheduleEntry, TimeSlot, Day, SessionType
)


class RefinementAgent(BaseAgent):
    """
    Uses LLM to suggest and apply schedule improvements.
    Focuses on reducing student conflicts through slot swaps.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
    
    def analyze_conflicts(
        self,
        proposal: TimetableProposal,
        student_conflicts: dict
    ) -> list[dict]:
        """
        Identify the highest-conflict time slots in the schedule.
        """
        # Group entries by slot
        slot_entries = defaultdict(list)
        for entry in proposal.entries:
            key = (entry.time_slot.day.value, entry.time_slot.hour)
            slot_entries[key].append(entry)
        
        # Calculate conflicts per slot
        slot_conflicts = []
        for (day, hour), entries in slot_entries.items():
            course_batches = [(e.course_code, e.batch_id) for e in entries]
            
            # Count conflicts in this slot
            conflict_count = 0
            for i, cb1 in enumerate(course_batches):
                for cb2 in course_batches[i+1:]:
                    key = (cb1, cb2) if cb1 < cb2 else (cb2, cb1)
                    conflict_count += student_conflicts.get(key, 0)
            
            if conflict_count > 0:
                slot_conflicts.append({
                    "day": day,
                    "hour": hour,
                    "courses": [f"{e.course_code}-{e.batch_id}" for e in entries],
                    "conflicts": conflict_count
                })
        
        # Sort by conflict count
        return sorted(slot_conflicts, key=lambda x: -x["conflicts"])
    
    def suggest_refinements(
        self,
        high_conflict_slots: list[dict],
        low_conflict_slots: list[dict],
        teacher_schedule: dict
    ) -> dict:
        """
        Use LLM to suggest specific schedule refinements.
        """
        prompt = f"""You are a timetable optimization expert. Analyze these conflict patterns and suggest improvements.

HIGH-CONFLICT TIME SLOTS (many students double-booked):
{chr(10).join(f"- {s['day']} {s['hour']}:00: {len(s['courses'])} courses, {s['conflicts']} student conflicts" for s in high_conflict_slots[:5])}

Sample courses in worst slot:
{', '.join(high_conflict_slots[0]['courses'][:6]) if high_conflict_slots else 'None'}

LOW-CONFLICT TIME SLOTS (good candidates for moving courses):
{chr(10).join(f"- {s['day']} {s['hour']}:00: {s['conflicts']} conflicts" for s in low_conflict_slots[:5])}

TASK: Suggest a refinement strategy to reduce student conflicts.

Respond with JSON:
{{
  "strategy": "brief description of the approach",
  "moves": [
    {{
      "action": "move|swap",
      "target": "course-batch to move",
      "from": "day hour",
      "to": "day hour",
      "reason": "why this helps"
    }}
  ],
  "expected_reduction": "estimated % reduction in conflicts",
  "risks": ["potential issues with these changes"]
}}"""

        self.log("Calling LLM for refinement suggestions...")
        response = self._call_llm(prompt, temperature=0.4)
        
        try:
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                self.log(f"LLM suggested {len(result.get('moves', []))} refinements")
                return result
        except Exception as e:
            self.log(f"Failed to parse LLM refinements: {e}")
        
        return {"error": "Failed to get refinements"}
    
    def apply_refinements(
        self,
        proposal: TimetableProposal,
        moves: list[dict],
        pair_student_count: dict
    ) -> tuple[TimetableProposal, int]:
        """
        Apply LLM-suggested refinements to the schedule.
        Actually moves entries to new slots and validates constraints.
        Returns (new_proposal, number_of_applied_moves).
        """
        entries = list(proposal.entries)
        applied = 0
        
        # Build occupancy maps
        teacher_schedule = defaultdict(set)  # (day, hour) -> set of teachers
        room_schedule = defaultdict(set)     # (day, hour) -> set of rooms
        
        for entry in entries:
            key = (entry.time_slot.day.value, entry.time_slot.hour)
            teacher_schedule[key].add(entry.teacher_name)
            room_schedule[key].add(entry.room_id)
        
        # Parse day name mapping
        day_map = {d.value: d for d in Day}
        
        for move in moves:
            action = move.get("action", "move")
            target = move.get("target", "")
            from_slot = move.get("from", "")
            to_slot = move.get("to", "")
            
            if not target or not to_slot:
                continue
            
            # Parse target course-batch (e.g., "EC401-B1")
            if "-" not in target:
                continue
            
            parts = target.rsplit("-", 1)
            if len(parts) != 2:
                continue
            
            course = parts[0]
            batch = parts[1]
            
            # Parse destination slot (e.g., "Thursday 14" or "Thu 14:00")
            to_parts = to_slot.replace(":00", "").split()
            if len(to_parts) < 2:
                continue
            
            to_day_str = to_parts[0]
            try:
                to_hour = int(to_parts[1])
            except ValueError:
                continue
            
            # Map day string to Day enum
            to_day = None
            for day_enum in Day:
                if day_enum.value.lower().startswith(to_day_str.lower()[:3]):
                    to_day = day_enum
                    break
            
            if not to_day:
                continue
            
            # Find matching entry
            for i, entry in enumerate(entries):
                if entry.course_code == course and entry.batch_id == batch:
                    new_slot_key = (to_day.value, to_hour)
                    
                    # Check if teacher is free in new slot
                    if entry.teacher_name in teacher_schedule[new_slot_key]:
                        self.log(f"Cannot move {target}: teacher busy at {to_slot}")
                        continue
                    
                    # Check if room is free in new slot
                    if entry.room_id in room_schedule[new_slot_key]:
                        # Try to find alternative room
                        alt_room = None
                        if entry.room_id.startswith("LAB"):
                            for r in range(1, 8):
                                if f"LAB{r}" not in room_schedule[new_slot_key]:
                                    alt_room = f"LAB{r}"
                                    break
                        else:
                            for r in range(1, 29):
                                if f"R{r}" not in room_schedule[new_slot_key]:
                                    alt_room = f"R{r}"
                                    break
                        
                        if not alt_room:
                            self.log(f"Cannot move {target}: no room at {to_slot}")
                            continue
                        
                        room_to_use = alt_room
                    else:
                        room_to_use = entry.room_id
                    
                    # Apply the move
                    old_slot_key = (entry.time_slot.day.value, entry.time_slot.hour)
                    
                    # Update occupancy maps
                    teacher_schedule[old_slot_key].discard(entry.teacher_name)
                    room_schedule[old_slot_key].discard(entry.room_id)
                    teacher_schedule[new_slot_key].add(entry.teacher_name)
                    room_schedule[new_slot_key].add(room_to_use)
                    
                    # Create new entry with updated slot
                    new_entry = ScheduleEntry(
                        course_code=entry.course_code,
                        batch_id=entry.batch_id,
                        teacher_name=entry.teacher_name,
                        room_id=room_to_use,
                        time_slot=TimeSlot(day=to_day, hour=to_hour),
                        session_type=entry.session_type,
                        student_count=entry.student_count
                    )
                    
                    entries[i] = new_entry
                    applied += 1
                    self.log(f"✅ Moved {target} from {entry.time_slot.day.value} {entry.time_slot.hour}:00 to {to_day.value} {to_hour}:00")
                    break
        
        self.log(f"Applied {applied}/{len(moves)} LLM-suggested moves")
        
        new_proposal = TimetableProposal(
            proposal_id=f"{proposal.proposal_id}_refined",
            entries=entries,
            algorithm_used="llm_refined",
            generation_time_ms=proposal.generation_time_ms
        )
        
        return new_proposal, applied
    
    def calculate_conflicts(
        self,
        proposal: TimetableProposal,
        pair_student_count: dict
    ) -> int:
        """Calculate total student conflicts for a proposal."""
        slot_courses = defaultdict(list)
        for entry in proposal.entries:
            key = (entry.time_slot.day.value, entry.time_slot.hour)
            slot_courses[key].append((entry.course_code, entry.batch_id))
        
        total = 0
        for cbs in slot_courses.values():
            for i, cb1 in enumerate(cbs):
                for cb2 in cbs[i+1:]:
                    key = (cb1, cb2) if cb1 < cb2 else (cb2, cb1)
                    total += pair_student_count.get(key, 0)
        return total
    
    def iterative_refinement(
        self,
        proposal: TimetableProposal,
        pair_student_count: dict,
        max_iterations: int = 3
    ) -> tuple[TimetableProposal, list[dict]]:
        """
        Perform multiple rounds of LLM-guided refinement.
        Actually applies changes and tracks improvement.
        """
        trace = []
        current = proposal
        initial_conflicts = self.calculate_conflicts(current, pair_student_count)
        
        self.log(f"Starting iterative refinement. Initial conflicts: {initial_conflicts}")
        
        for iteration in range(max_iterations):
            self.log(f"Refinement iteration {iteration + 1}/{max_iterations}")
            
            # Analyze current conflicts
            high_conflicts = self.analyze_conflicts(current, pair_student_count)
            current_total = self.calculate_conflicts(current, pair_student_count)
            
            if not high_conflicts or high_conflicts[0]["conflicts"] < 50:
                self.log("Conflicts below threshold, stopping refinement")
                break
            
            # Find low-conflict slots as move targets
            all_slots_by_conflict = self.analyze_conflicts(current, pair_student_count)
            low_conflicts = [s for s in all_slots_by_conflict if s["conflicts"] < 100][-10:]
            
            # If no low-conflict slots, create list of empty slots
            if not low_conflicts:
                low_conflicts = [{"day": d.value, "hour": h, "conflicts": 0} 
                                for d in Day for h in range(10, 18)]
            
            # Get LLM suggestions
            suggestions = self.suggest_refinements(
                high_conflicts[:5],
                low_conflicts[:5],
                {}
            )
            
            moves = suggestions.get("moves", [])
            
            # Actually apply the moves
            if moves:
                current, applied = self.apply_refinements(current, moves, pair_student_count)
                new_conflicts = self.calculate_conflicts(current, pair_student_count)
                improvement = current_total - new_conflicts
            else:
                applied = 0
                new_conflicts = current_total
                improvement = 0
            
            trace.append({
                "iteration": iteration + 1,
                "conflicts_before": current_total,
                "conflicts_after": new_conflicts,
                "improvement": improvement,
                "moves_applied": applied,
                "suggestions": suggestions
            })
            
            self.log(f"Iteration {iteration + 1}: {current_total} → {new_conflicts} conflicts (Δ{improvement})")
            
            if suggestions.get("error") or applied == 0:
                self.log("No more improvements possible, stopping")
                break
        
        final_conflicts = self.calculate_conflicts(current, pair_student_count)
        total_improvement = initial_conflicts - final_conflicts
        self.log(f"Refinement complete: {initial_conflicts} → {final_conflicts} (reduced by {total_improvement})")
        
        return current, trace

