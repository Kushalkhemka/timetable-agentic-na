"""
Constraint Agent - Extracts scheduling constraints from the problem data.
"""
import json
from .base_agent import BaseAgent
from models.data_models import Constraint, Course, Teacher, SchedulingConfig
from typing import Optional


class ConstraintAgent(BaseAgent):
    """
    Extracts hard and soft constraints for the timetable scheduling problem.
    Following PlanGEN framework: analyzes problem structure to identify constraints.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
    
    def extract_constraints(
        self,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig
    ) -> list[Constraint]:
        """
        Extract all constraints from the scheduling problem.
        Uses LLM to identify implicit constraints and generates explicit ones programmatically.
        """
        constraints = []
        
        # 1. Generate hard constraints (programmatic)
        constraints.extend(self._generate_hard_constraints(courses, teachers, config))
        
        # 2. Generate soft constraints (programmatic with some LLM enhancement)
        constraints.extend(self._generate_soft_constraints(courses, teachers, config))
        
        self.log(f"Extracted {len(constraints)} total constraints")
        return constraints
    
    def _generate_hard_constraints(
        self,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig
    ) -> list[Constraint]:
        """Generate hard constraints that must be satisfied."""
        constraints = []
        
        # HC1: No teacher can teach two different sessions at the same time
        for teacher_name, teacher in teachers.items():
            constraints.append(Constraint(
                constraint_id=f"HC_TEACHER_{teacher_name.replace(' ', '_').replace('.', '')}",
                constraint_type="hard",
                description=f"Teacher {teacher_name} cannot be scheduled for multiple sessions at the same time slot",
                entities_involved=[teacher_name] + teacher.courses
            ))
        
        # HC2: No room can have two sessions at the same time
        for room in config.generate_all_rooms():
            constraints.append(Constraint(
                constraint_id=f"HC_ROOM_{room.room_id}",
                constraint_type="hard",
                description=f"Room {room.room_id} cannot host multiple sessions at the same time slot",
                entities_involved=[room.room_id]
            ))
        
        # HC3: Each course-batch must get exactly the required hours
        for code, course in courses.items():
            for batch_id in course.batches:
                constraints.append(Constraint(
                    constraint_id=f"HC_HOURS_{code}_{batch_id}",
                    constraint_type="hard",
                    description=f"Course {code} batch {batch_id} must have exactly {course.total_hours} hours scheduled ({course.theory_hours}hr theory + {course.lab_hours}hr lab)",
                    entities_involved=[code, batch_id]
                ))
        
        # HC4: Same batch of a course cannot have overlapping sessions
        for code, course in courses.items():
            for batch_id in course.batches:
                constraints.append(Constraint(
                    constraint_id=f"HC_BATCH_OVERLAP_{code}_{batch_id}",
                    constraint_type="hard",
                    description=f"Course {code} batch {batch_id} cannot have two sessions at the same time",
                    entities_involved=[code, batch_id]
                ))
        
        # HC5: Room capacity constraint
        constraints.append(Constraint(
            constraint_id="HC_ROOM_CAPACITY",
            constraint_type="hard",
            description=f"Each session must be assigned to a room with capacity >= student count (max capacity: {config.room_capacity})",
            entities_involved=["all_rooms"]
        ))
        
        # HC6: Time slot bounds
        constraints.append(Constraint(
            constraint_id="HC_TIME_BOUNDS",
            constraint_type="hard",
            description=f"All sessions must be scheduled within {config.start_hour}:00 - {config.end_hour}:00 on weekdays",
            entities_involved=["all_sessions"]
        ))
        
        return constraints
    
    def analyze_with_llm(
        self,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig,
        conflict_pairs: list[tuple] = None
    ) -> dict:
        """
        Use LLM to analyze the scheduling problem and discover implicit constraints.
        Returns insights about bottlenecks, high-conflict pairs, and recommendations.
        """
        # Prepare problem summary
        total_sessions = sum(c.total_hours * len(c.batches) for c in courses.values())
        total_slots = config.total_slots * config.num_rooms
        
        # Find teachers with most courses
        teacher_loads = {}
        for name, teacher in teachers.items():
            teacher_loads[name] = len(teacher.courses)
        busiest_teachers = sorted(teacher_loads.items(), key=lambda x: -x[1])[:5]
        
        # Format conflict pairs if provided
        conflict_info = ""
        if conflict_pairs:
            conflict_info = f"""
HIGH-CONFLICT COURSE PAIRS (share many students):
{chr(10).join(f'  - {p[0]} and {p[1]}: {p[2]} shared students' for p in conflict_pairs[:10])}
"""
        
        prompt = f"""You are a scheduling constraint expert. Analyze this university timetabling problem and identify implicit constraints and bottlenecks.

PROBLEM DATA:
- Courses: {len(courses)}
- Teachers: {len(teachers)}
- Total sessions to schedule: {total_sessions}
- Available slots: {total_slots} (utilization: {total_sessions/total_slots*100:.1f}%)
- Days: 5 (Monday-Friday)
- Hours per day: 8 (10:00-18:00)
- Rooms: {config.num_rooms} (including labs)

BUSIEST TEACHERS:
{chr(10).join(f'  - {t[0]}: {t[1]} courses' for t in busiest_teachers)}
{conflict_info}
TASK: Identify implicit constraints and scheduling insights.

Respond with JSON:
{{
  "bottleneck_teachers": ["teacher names who may cause conflicts"],
  "critical_course_pairs": ["course pairs that MUST be on different days"],
  "recommended_priority_order": ["which course types to schedule first"],
  "potential_issues": ["issues that may prevent 100% scheduling"],
  "suggested_constraints": [
    {{"type": "hard|soft", "description": "constraint description"}}
  ]
}}"""

        self.log("Calling LLM for constraint analysis...")
        response = self._call_llm(prompt, temperature=0.3)
        
        try:
            # Parse JSON response
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                self.log(f"LLM identified {len(result.get('suggested_constraints', []))} implicit constraints")
                return result
        except Exception as e:
            self.log(f"Failed to parse LLM response: {e}")
        
        return {"error": "Failed to analyze", "raw_response": response[:200]}
    
    def _generate_soft_constraints(
        self,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig
    ) -> list[Constraint]:
        """Generate soft constraints (preferences) that improve quality."""
        constraints = []
        
        # SC1: Distribute sessions across days (avoid all sessions on one day)
        for code, course in courses.items():
            for batch_id in course.batches:
                constraints.append(Constraint(
                    constraint_id=f"SC_SPREAD_{code}_{batch_id}",
                    constraint_type="soft",
                    description=f"Prefer spreading {code} batch {batch_id} sessions across different days",
                    entities_involved=[code, batch_id]
                ))
        
        # SC2: Teacher workload balance per day
        for teacher_name in teachers.keys():
            constraints.append(Constraint(
                constraint_id=f"SC_WORKLOAD_{teacher_name.replace(' ', '_').replace('.', '')}",
                constraint_type="soft",
                description=f"Prefer balanced daily teaching load for {teacher_name}",
                entities_involved=[teacher_name]
            ))
        
        # SC3: Avoid back-to-back classes for teachers (when possible)
        constraints.append(Constraint(
            constraint_id="SC_TEACHER_BREAKS",
            constraint_type="soft",
            description="Prefer giving teachers breaks between consecutive classes when possible",
            entities_involved=list(teachers.keys())
        ))
        
        # SC4: Prefer morning slots for theory
        constraints.append(Constraint(
            constraint_id="SC_THEORY_MORNING",
            constraint_type="soft",
            description="Prefer scheduling theory sessions in morning slots (before 2pm)",
            entities_involved=["theory_sessions"]
        ))
        
        # SC5: Room utilization efficiency
        constraints.append(Constraint(
            constraint_id="SC_ROOM_UTILIZATION",
            constraint_type="soft",
            description="Prefer efficient room utilization (match batch size to room capacity)",
            entities_involved=["all_rooms"]
        ))
        
        return constraints
    
    def analyze_constraint_density(
        self,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig
    ) -> dict:
        """
        Analyze the constraint density to help Selection Agent choose algorithm.
        Returns metrics about problem complexity.
        """
        total_sessions = sum(c.total_hours * len(c.batches) for c in courses.values())
        total_slots = config.total_slots * config.num_rooms
        
        # Calculate teacher conflict potential
        teacher_loads = {t: len(courses) for t, courses in 
                        {t.name: t.courses for t in teachers.values()}.items()}
        max_teacher_load = max(teacher_loads.values()) if teacher_loads else 0
        
        # Density metrics
        slot_utilization = total_sessions / total_slots if total_slots > 0 else 0
        
        return {
            "total_sessions": total_sessions,
            "available_slots": total_slots,
            "slot_utilization_ratio": round(slot_utilization, 3),
            "num_teachers": len(teachers),
            "max_teacher_load": max_teacher_load,
            "avg_batches_per_course": sum(len(c.batches) for c in courses.values()) / len(courses) if courses else 0,
            "complexity_score": self._calculate_complexity_score(
                total_sessions, total_slots, len(teachers), max_teacher_load
            )
        }
    
    def _calculate_complexity_score(
        self,
        total_sessions: int,
        total_slots: int,
        num_teachers: int,
        max_teacher_load: int
    ) -> str:
        """Calculate overall problem complexity: low, medium, high."""
        utilization = total_sessions / total_slots if total_slots > 0 else 1
        
        if utilization > 0.7 or max_teacher_load > 10:
            return "high"
        elif utilization > 0.4 or max_teacher_load > 6:
            return "medium"
        else:
            return "low"
    
    def get_constraint_summary(self, constraints: list[Constraint]) -> str:
        """Generate a human-readable summary of constraints."""
        hard = [c for c in constraints if c.constraint_type == "hard"]
        soft = [c for c in constraints if c.constraint_type == "soft"]
        
        summary = f"=== Constraint Summary ===\n"
        summary += f"Hard Constraints: {len(hard)}\n"
        summary += f"Soft Constraints: {len(soft)}\n\n"
        
        summary += "Key Hard Constraints:\n"
        for c in hard[:5]:
            summary += f"  - {c.description}\n"
        
        summary += "\nKey Soft Constraints:\n"
        for c in soft[:5]:
            summary += f"  - {c.description}\n"
        
        return summary
