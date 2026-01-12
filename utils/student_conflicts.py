"""
Student Conflict Matrix - Builds conflict relationships between course-batches.
Used by scheduler to ensure students with multiple courses don't have overlapping classes.
"""
import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional


class StudentConflictMatrix:
    """
    Tracks which course-batches share students and therefore CANNOT be scheduled
    at the same time slot.
    """
    
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        # (course, batch) -> set of (course, batch) that conflict
        self.conflicts: dict[tuple, set] = defaultdict(set)
        # (day, hour) -> set of (course, batch) scheduled at this slot
        self.slot_schedule: dict[tuple, set] = defaultdict(set)
        
        self._load_conflicts()
    
    def _load_conflicts(self):
        """Load student allocations and build conflict matrix."""
        student_file = self.data_dir / "student_allocations_aggregated.csv"
        
        if not student_file.exists():
            print("[StudentConflictMatrix] Warning: student allocations file not found")
            return
        
        student_courses = defaultdict(list)
        
        with open(student_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                courses = row['Allocated Courses'].replace('"', '').split(', ')
                batches = row['Batches'].replace('"', '').split(', ')
                
                course_batches = []
                for course, batch in zip(courses, batches):
                    course_batches.append((course.strip(), batch.strip()))
                
                # All pairs of courses for this student conflict with each other
                for i, cb1 in enumerate(course_batches):
                    for cb2 in course_batches[i+1:]:
                        self.conflicts[cb1].add(cb2)
                        self.conflicts[cb2].add(cb1)
        
        print(f"[StudentConflictMatrix] Loaded conflicts for {len(self.conflicts)} course-batches")
    
    def check_slot_available(self, course: str, batch: str, day: str, hour: int) -> bool:
        """
        Check if scheduling (course, batch) at (day, hour) would cause student conflicts.
        Returns True if no conflicts, False if there would be a conflict.
        """
        cb = (course, batch)
        slot_key = (day, hour)
        
        # Get courses already scheduled at this slot
        courses_at_slot = self.slot_schedule[slot_key]
        
        # Check if any scheduled course conflicts with new one
        conflicting_courses = self.conflicts.get(cb, set())
        
        for scheduled_cb in courses_at_slot:
            if scheduled_cb in conflicting_courses:
                return False  # Conflict found!
        
        return True  # No conflict
    
    def mark_scheduled(self, course: str, batch: str, day: str, hour: int):
        """Mark a course-batch as scheduled at a slot."""
        cb = (course, batch)
        slot_key = (day, hour)
        self.slot_schedule[slot_key].add(cb)
    
    def unmark_scheduled(self, course: str, batch: str, day: str, hour: int):
        """Remove a course-batch from a slot (for rescheduling)."""
        cb = (course, batch)
        slot_key = (day, hour)
        self.slot_schedule[slot_key].discard(cb)
    
    def get_conflicts_count(self, course: str, batch: str) -> int:
        """Get number of course-batches that conflict with this one."""
        return len(self.conflicts.get((course, batch), set()))
    
    def reset_schedule(self):
        """Clear all scheduled slots for fresh scheduling."""
        self.slot_schedule = defaultdict(set)
