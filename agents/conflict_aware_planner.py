"""
Conflict-Aware Planner - Minimizes student conflicts while maximizing coverage.
Uses graph-coloring inspired approach to find optimal slot assignments.
"""
import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional
import time

from models.data_models import (
    Course, Teacher, SchedulingConfig, TimeSlot, Room,
    ScheduleEntry, TimetableProposal, SessionType, Day
)


class ConflictAwarePlanner:
    """
    Improved planner that:
    1. Sorts courses by conflict density (hardest first)
    2. For each session, picks the slot with MINIMUM existing conflicts
    3. Uses all available rooms to spread courses
    """
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.student_conflicts = defaultdict(set)  # (course, batch) -> set of conflicting (course, batch)
        self.conflict_count = defaultdict(int)  # (course, batch) -> number of conflicts
        self._load_student_conflicts()
    
    def _load_student_conflicts(self):
        """Load student conflict matrix WITH student counts per pair."""
        student_file = self.data_dir / "student_allocations_aggregated.csv"
        
        # First: count students sharing each course-batch pair
        self.pair_student_count = defaultdict(int)  # (cb1, cb2) -> student count
        
        with open(student_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                courses = row['Allocated Courses'].replace('"', '').split(', ')
                batches = row['Batches'].replace('"', '').split(', ')
                
                cbs = [(c.strip(), b.strip()) for c, b in zip(courses, batches)]
                for i, cb1 in enumerate(cbs):
                    for cb2 in cbs[i+1:]:
                        self.student_conflicts[cb1].add(cb2)
                        self.student_conflicts[cb2].add(cb1)
                        # Count students affected by this pair
                        key = (cb1, cb2) if cb1 < cb2 else (cb2, cb1)
                        self.pair_student_count[key] += 1
        
        # Count WEIGHTED conflicts per course-batch (sum of all students in conflicting pairs)
        for cb in self.student_conflicts:
            weighted_count = 0
            for conflict_cb in self.student_conflicts[cb]:
                key = (cb, conflict_cb) if cb < conflict_cb else (conflict_cb, cb)
                weighted_count += self.pair_student_count[key]
            self.conflict_count[cb] = weighted_count
        
        print(f"[ConflictAwarePlanner] Loaded {len(self.student_conflicts)} course-batches")
        print(f"[ConflictAwarePlanner] Total conflict pairs: {len(self.pair_student_count)}")
    
    def generate_proposal(
        self,
        courses: dict[str, Course],
        teachers: dict[str, Teacher],
        config: SchedulingConfig
    ) -> TimetableProposal:
        """Generate schedule minimizing student conflicts."""
        start_time = time.time()
        
        # Tracking structures
        entries = []
        teacher_schedule = defaultdict(set)  # (day, hour) -> set of teachers
        room_schedule = defaultdict(bool)  # (day, hour, room) -> bool
        slot_courses = defaultdict(set)  # (day, hour) -> set of (course, batch)
        
        # Create rooms
        regular_rooms = [f"R{i}" for i in range(1, 29)]  # R1-R28
        lab_rooms = [f"LAB{i}" for i in range(1, 8)]  # LAB1-LAB7
        
        # Generate all slots
        days = list(Day)
        hours = list(range(10, 18))
        
        # Build list of (course, batch, type, hours_needed)
        sessions_to_schedule = []
        for code, course in courses.items():
            for batch_idx, batch_id in enumerate(course.batches):
                cb = (code, batch_id)
                conflict_score = self.conflict_count.get(cb, 0)
                
                if course.theory_hours > 0:
                    sessions_to_schedule.append({
                        "course": code,
                        "batch": batch_id,
                        "type": "theory",
                        "hours": course.theory_hours,
                        "teacher": course.teacher_assignments.get(batch_id, "TBA"),
                        "students": course.batch_sizes[batch_idx] if batch_idx < len(course.batch_sizes) else 60,
                        "conflicts": conflict_score
                    })
                
                if course.lab_hours > 0:
                    sessions_to_schedule.append({
                        "course": code,
                        "batch": batch_id,
                        "type": "lab",
                        "hours": course.lab_hours,
                        "teacher": course.teacher_assignments.get(batch_id, "TBA"),
                        "students": course.batch_sizes[batch_idx] if batch_idx < len(course.batch_sizes) else 60,
                        "conflicts": conflict_score
                    })
        
        # SORT BY CONFLICTS (most constrained first - graph coloring heuristic)
        sessions_to_schedule.sort(key=lambda x: -x["conflicts"])
        
        print(f"[ConflictAwarePlanner] Scheduling {len(sessions_to_schedule)} session groups...")
        print(f"[ConflictAwarePlanner] Most conflicts: {sessions_to_schedule[0]['course']}-{sessions_to_schedule[0]['batch']} with {sessions_to_schedule[0]['conflicts']} conflicts")
        
        def count_slot_conflicts(day, hour, course, batch):
            """Count how many STUDENTS would be double-booked."""
            cb = (course, batch)
            student_conflicts = 0
            for scheduled_cb in slot_courses[(day, hour)]:
                if scheduled_cb in self.student_conflicts.get(cb, set()):
                    # Get actual student count for this conflict pair
                    key = (cb, scheduled_cb) if cb < scheduled_cb else (scheduled_cb, cb)
                    student_conflicts += self.pair_student_count.get(key, 0)
            return student_conflicts
        
        def find_best_slot(course, batch, teacher, session_type, rooms):
            """Find the slot with MINIMUM conflicts for this course-batch."""
            best_slot = None
            best_room = None
            best_conflict_count = float('inf')
            
            for day in days:
                for hour in hours:
                    # Check teacher availability
                    if teacher in teacher_schedule[(day, hour)]:
                        continue
                    
                    # Count conflicts
                    conflicts = count_slot_conflicts(day, hour, course, batch)
                    
                    # Find available room
                    for room in rooms:
                        if not room_schedule[(day, hour, room)]:
                            if conflicts < best_conflict_count:
                                best_conflict_count = conflicts
                                best_slot = (day, hour)
                                best_room = room
                                
                                # If zero conflicts, use it immediately
                                if conflicts == 0:
                                    return best_slot, best_room, best_conflict_count
                            break  # Found a room, check next slot
            
            return best_slot, best_room, best_conflict_count
        
        total_conflicts = 0
        scheduled_count = 0
        
        for session in sessions_to_schedule:
            course = session["course"]
            batch = session["batch"]
            teacher = session["teacher"]
            session_type = session["type"]
            hours_needed = session["hours"]
            students = session["students"]
            
            rooms = lab_rooms if session_type == "lab" else regular_rooms
            
            # Schedule required hours
            for _ in range(hours_needed):
                slot, room, conflicts = find_best_slot(course, batch, teacher, session_type, rooms)
                
                if slot is None:
                    break  # No slot available
                
                day, hour = slot
                
                # Create entry
                entry = ScheduleEntry(
                    course_code=course,
                    batch_id=batch,
                    teacher_name=teacher,
                    room_id=room,
                    time_slot=TimeSlot(day=day, hour=hour),
                    session_type=SessionType.LAB if session_type == "lab" else SessionType.THEORY,
                    student_count=students
                )
                entries.append(entry)
                
                # Update tracking
                teacher_schedule[(day, hour)].add(teacher)
                room_schedule[(day, hour, room)] = True
                slot_courses[(day, hour)].add((course, batch))
                
                total_conflicts += conflicts
                scheduled_count += 1
        
        print(f"[ConflictAwarePlanner] Scheduled {scheduled_count} sessions")
        print(f"[ConflictAwarePlanner] Total student conflict instances: {total_conflicts}")
        
        return TimetableProposal(
            proposal_id=f"conflict_aware_{int(time.time())}",
            entries=entries,
            algorithm_used="conflict_aware",
            generation_time_ms=(time.time() - start_time) * 1000
        )


def run_conflict_aware_scheduler(data_dir: str = "."):
    """Run the conflict-aware scheduler."""
    from utils.data_loader import DataLoader
    from models.data_models import SchedulingConfig
    
    print("="*60)
    print("🧠 Conflict-Aware Timetable Scheduler")
    print("="*60)
    
    # Load data
    loader = DataLoader(data_dir)
    loader.load_all()
    courses = loader.courses
    teachers = loader.teachers
    config = SchedulingConfig(num_rooms=30, room_capacity=90)
    
    print(f"Loaded {len(courses)} courses, {len(teachers)} teachers")
    
    # Run planner
    planner = ConflictAwarePlanner(data_dir)
    proposal = planner.generate_proposal(courses, teachers, config)
    
    print(f"\n📊 Results:")
    print(f"   Sessions: {len(proposal.entries)}")
    print(f"   Time: {proposal.generation_time_ms:.0f}ms")
    
    # Save output
    output_dir = Path(data_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "timetable.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Day", "Hour", "Course", "Batch", "Teacher", "Room", "Type", "Students"])
        
        for entry in sorted(proposal.entries, key=lambda e: (
            list(Day).index(e.time_slot.day), e.time_slot.hour
        )):
            writer.writerow([
                entry.time_slot.day.value,
                f"{entry.time_slot.hour}:00-{entry.time_slot.hour+1}:00",
                entry.course_code,
                entry.batch_id,
                entry.teacher_name,
                entry.room_id,
                entry.session_type.value,
                entry.student_count
            ])
    
    print(f"\n✅ Saved to {output_dir / 'timetable.csv'}")
    
    return proposal


if __name__ == "__main__":
    run_conflict_aware_scheduler(".")
