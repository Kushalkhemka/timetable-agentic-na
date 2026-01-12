"""
Verification Agent - Validates timetable proposals against constraints.
"""
from typing import Optional
from collections import defaultdict

from .base_agent import BaseAgent
from models.data_models import (
    Course, Teacher, SchedulingConfig, TimetableProposal, 
    VerificationResult, Constraint
)


class VerificationAgent(BaseAgent):
    """
    Validates timetable proposals and provides feedback.
    Following PlanGEN: scores proposals and provides natural language feedback.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
    
    def verify(
        self,
        proposal: TimetableProposal,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig,
        constraints: list[Constraint]
    ) -> VerificationResult:
        """
        Verify a timetable proposal against all constraints.
        Returns score, conflicts, and feedback.
        """
        conflicts = []
        
        # Check hard constraints
        conflicts.extend(self._check_teacher_conflicts(proposal))
        conflicts.extend(self._check_room_conflicts(proposal))
        conflicts.extend(self._check_coverage(proposal, courses))
        conflicts.extend(self._check_time_bounds(proposal, config))
        
        # Check soft constraints (lower severity)
        soft_issues = self._check_soft_constraints(proposal, courses, config)
        
        # Calculate score
        hard_conflict_count = len([c for c in conflicts if c["severity"] == "hard"])
        soft_conflict_count = len(soft_issues)
        
        # Score: start at 1.0, deduct for conflicts
        score = 1.0
        score -= hard_conflict_count * 0.1  # Heavy penalty for hard conflicts
        score -= soft_conflict_count * 0.02  # Light penalty for soft issues
        score = max(0.0, min(1.0, score))
        
        # Check coverage
        total_required = sum(c.total_hours * len(c.batches) for c in courses.values())
        coverage = len(proposal.entries) / total_required if total_required > 0 else 0
        score *= coverage  # Scale by coverage
        
        # Generate feedback
        feedback = self._generate_feedback(conflicts, soft_issues, coverage, proposal)
        suggestions = self._generate_suggestions(conflicts, soft_issues)
        
        is_valid = hard_conflict_count == 0 and coverage >= 0.95
        
        result = VerificationResult(
            proposal_id=proposal.proposal_id,
            is_valid=is_valid,
            score=round(score, 3),
            conflicts=conflicts + soft_issues,
            feedback=feedback,
            suggestions=suggestions
        )
        
        self.log(f"Verification: score={result.score}, valid={is_valid}, conflicts={len(conflicts)}")
        return result
    
    def _check_teacher_conflicts(self, proposal: TimetableProposal) -> list[dict]:
        """Check if any teacher is double-booked."""
        conflicts = []
        
        # Group entries by (day, hour)
        slot_entries = defaultdict(list)
        for entry in proposal.entries:
            key = (entry.time_slot.day.value, entry.time_slot.hour)
            slot_entries[key].append(entry)
        
        # Check for teacher conflicts
        for slot_key, entries in slot_entries.items():
            teacher_counts = defaultdict(list)
            for entry in entries:
                teacher_counts[entry.teacher_name].append(entry)
            
            for teacher, teacher_entries in teacher_counts.items():
                if len(teacher_entries) > 1:
                    conflicts.append({
                        "type": "teacher_conflict",
                        "severity": "hard",
                        "description": f"{teacher} is scheduled for {len(teacher_entries)} classes at {slot_key[0]} {slot_key[1]}:00",
                        "entries": [
                            f"{e.course_code}-{e.batch_id}" for e in teacher_entries
                        ]
                    })
        
        return conflicts
    
    def _check_room_conflicts(self, proposal: TimetableProposal) -> list[dict]:
        """Check if any room is double-booked."""
        conflicts = []
        
        # Group entries by (day, hour, room)
        room_entries = defaultdict(list)
        for entry in proposal.entries:
            key = (entry.time_slot.day.value, entry.time_slot.hour, entry.room_id)
            room_entries[key].append(entry)
        
        for key, entries in room_entries.items():
            if len(entries) > 1:
                conflicts.append({
                    "type": "room_conflict",
                    "severity": "hard",
                    "description": f"Room {key[2]} has {len(entries)} classes at {key[0]} {key[1]}:00",
                    "entries": [f"{e.course_code}-{e.batch_id}" for e in entries]
                })
        
        return conflicts
    
    def _check_coverage(self, proposal: TimetableProposal, courses: dict[str, Course]) -> list[dict]:
        """Check if all courses have sufficient hours scheduled."""
        conflicts = []
        
        # Count scheduled hours per course-batch
        scheduled = defaultdict(lambda: {"theory": 0, "lab": 0})
        for entry in proposal.entries:
            key = (entry.course_code, entry.batch_id)
            if entry.session_type.value == "Theory":
                scheduled[key]["theory"] += 1
            else:
                scheduled[key]["lab"] += 1
        
        # Check against requirements
        for code, course in courses.items():
            for batch_id in course.batches:
                key = (code, batch_id)
                theory_scheduled = scheduled[key]["theory"]
                lab_scheduled = scheduled[key]["lab"]
                
                theory_missing = course.theory_hours - theory_scheduled
                lab_missing = course.lab_hours - lab_scheduled
                
                if theory_missing > 0:
                    conflicts.append({
                        "type": "incomplete_coverage",
                        "severity": "hard",
                        "description": f"{code}-{batch_id} missing {theory_missing} theory hour(s)",
                        "entries": []
                    })
                
                if lab_missing > 0:
                    conflicts.append({
                        "type": "incomplete_coverage",
                        "severity": "hard",
                        "description": f"{code}-{batch_id} missing {lab_missing} lab hour(s)",
                        "entries": []
                    })
        
        return conflicts
    
    def _check_time_bounds(self, proposal: TimetableProposal, config: SchedulingConfig) -> list[dict]:
        """Check if all entries are within valid time bounds."""
        conflicts = []
        valid_days = [d.value for d in config.days]
        
        for entry in proposal.entries:
            if entry.time_slot.day.value not in valid_days:
                conflicts.append({
                    "type": "invalid_day",
                    "severity": "hard",
                    "description": f"{entry.course_code}-{entry.batch_id} scheduled on invalid day {entry.time_slot.day.value}",
                    "entries": [f"{entry.course_code}-{entry.batch_id}"]
                })
            
            if entry.time_slot.hour < config.start_hour or entry.time_slot.hour >= config.end_hour:
                conflicts.append({
                    "type": "invalid_time",
                    "severity": "hard",
                    "description": f"{entry.course_code}-{entry.batch_id} scheduled at invalid time {entry.time_slot.hour}:00",
                    "entries": [f"{entry.course_code}-{entry.batch_id}"]
                })
        
        return conflicts
    
    def _check_soft_constraints(
        self,
        proposal: TimetableProposal,
        courses: dict[str, Course],
        config: SchedulingConfig
    ) -> list[dict]:
        """Check soft constraints (preferences)."""
        issues = []
        
        # Check for course sessions on same day (clustering)
        course_batch_days = defaultdict(set)
        for entry in proposal.entries:
            key = (entry.course_code, entry.batch_id)
            course_batch_days[key].add(entry.time_slot.day.value)
        
        for (code, batch), days in course_batch_days.items():
            course = courses.get(code)
            if course and len(days) < min(course.total_hours, 3):
                # Sessions too clustered
                issues.append({
                    "type": "poor_distribution",
                    "severity": "soft",
                    "description": f"{code}-{batch} has all {course.total_hours} sessions on only {len(days)} day(s)",
                    "entries": []
                })
        
        # Check teacher workload per day
        teacher_daily_load = defaultdict(lambda: defaultdict(int))
        for entry in proposal.entries:
            teacher_daily_load[entry.teacher_name][entry.time_slot.day.value] += 1
        
        for teacher, daily in teacher_daily_load.items():
            for day, count in daily.items():
                if count > 6:  # More than 6 hours on one day
                    issues.append({
                        "type": "teacher_overload",
                        "severity": "soft",
                        "description": f"{teacher} has {count} hours on {day}",
                        "entries": []
                    })
        
        return issues
    
    def _generate_feedback(
        self,
        conflicts: list[dict],
        soft_issues: list[dict],
        coverage: float,
        proposal: TimetableProposal
    ) -> str:
        """Generate human-readable feedback."""
        feedback = []
        
        if not conflicts and coverage >= 0.95:
            feedback.append("✅ Schedule is valid with good coverage.")
        else:
            if conflicts:
                feedback.append(f"❌ Found {len(conflicts)} hard constraint violations:")
                for c in conflicts[:5]:
                    feedback.append(f"  - {c['description']}")
                if len(conflicts) > 5:
                    feedback.append(f"  ... and {len(conflicts) - 5} more")
            
            if coverage < 0.95:
                feedback.append(f"⚠️ Coverage is only {coverage*100:.1f}% of required sessions")
        
        if soft_issues:
            feedback.append(f"\n📝 {len(soft_issues)} soft constraint issues (preferences):")
            for s in soft_issues[:3]:
                feedback.append(f"  - {s['description']}")
        
        return "\n".join(feedback)
    
    def _generate_suggestions(self, conflicts: list[dict], soft_issues: list[dict]) -> list[str]:
        """Generate actionable suggestions."""
        suggestions = []
        
        conflict_types = set(c["type"] for c in conflicts)
        
        if "teacher_conflict" in conflict_types:
            suggestions.append("Reschedule conflicting teacher sessions to different time slots")
        
        if "room_conflict" in conflict_types:
            suggestions.append("Assign conflicting sessions to different rooms")
        
        if "incomplete_coverage" in conflict_types:
            suggestions.append("Find additional slots for unscheduled sessions")
        
        if any(s["type"] == "poor_distribution" for s in soft_issues):
            suggestions.append("Spread course sessions across more days for better learning")
        
        return suggestions
    
    def get_llm_feedback(
        self,
        proposal: TimetableProposal,
        conflicts: list[dict],
        student_conflicts: int,
        courses: dict[str, Course]
    ) -> dict:
        """
        Use LLM to analyze verification results and provide intelligent feedback.
        Explains WHY conflicts occurred and suggests specific fixes.
        """
        import json
        
        # Prepare conflict summary
        conflict_summary = []
        for c in conflicts[:10]:
            conflict_summary.append(f"- {c['type']}: {c['description']}")
        
        # Calculate stats
        total_sessions = len(proposal.entries)
        total_required = sum(c.total_hours * len(c.batches) for c in courses.values())
        coverage = total_sessions / total_required * 100 if total_required > 0 else 0
        
        prompt = f"""You are a timetable verification expert. Analyze these scheduling results and provide feedback.

SCHEDULE RESULTS:
- Sessions scheduled: {total_sessions}/{total_required} ({coverage:.1f}% coverage)
- Student conflicts: {student_conflicts} (students double-booked)
- Hard constraint violations: {len([c for c in conflicts if c.get('severity') == 'hard'])}

SPECIFIC ISSUES:
{chr(10).join(conflict_summary) if conflict_summary else "No major issues detected"}

TASK: Analyze these results and provide actionable feedback.

Respond with JSON:
{{
  "overall_assessment": "good|acceptable|needs_improvement|critical",
  "root_causes": ["list of root causes for the issues"],
  "specific_fixes": [
    {{"issue": "description", "fix": "how to fix it"}}
  ],
  "priority_actions": ["top 3 actions to improve the schedule"],
  "estimated_improvement": "how much could conflicts be reduced with fixes"
}}"""

        self.log("Calling LLM for verification feedback...")
        response = self._call_llm(prompt, temperature=0.3)
        
        try:
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                self.log(f"LLM assessment: {result.get('overall_assessment', 'unknown')}")
                return result
        except Exception as e:
            self.log(f"Failed to parse LLM feedback: {e}")
        
        return {"error": "Failed to get feedback", "raw_response": response[:200]}
    
    def suggest_improvements_with_llm(
        self,
        high_conflict_slots: list[dict],
        available_slots: list[dict]
    ) -> list[dict]:
        """
        Use LLM to suggest specific slot swaps to reduce conflicts.
        """
        import json
        
        prompt = f"""You are a timetable optimization expert. Suggest slot swaps to reduce student conflicts.

HIGH-CONFLICT SLOTS (too many overlapping courses):
{chr(10).join(f"- {s['day']} {s['hour']}:00: {s['courses']} ({s['conflicts']} conflicts)" for s in high_conflict_slots[:8])}

AVAILABLE LOW-CONFLICT SLOTS:
{chr(10).join(f"- {s['day']} {s['hour']}:00 ({s['room']})" for s in available_slots[:10])}

TASK: Suggest 3-5 specific swaps to reduce conflicts.

Respond with JSON:
{{
  "suggested_swaps": [
    {{
      "move_course": "course-batch to move",
      "from_slot": "current day/time",
      "to_slot": "suggested day/time",
      "expected_reduction": "estimated conflict reduction"
    }}
  ],
  "reasoning": "brief explanation of the strategy"
}}"""

        self.log("Calling LLM for improvement suggestions...")
        response = self._call_llm(prompt, temperature=0.4)
        
        try:
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                return result.get("suggested_swaps", [])
        except Exception as e:
            self.log(f"Failed to parse LLM suggestions: {e}")
        
        return []
