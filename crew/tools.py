"""
Custom Tools for Timetable Scheduling Crew.
These tools allow agents to check constraints, assign slots, and verify schedules.
"""
import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional
from crewai.tools import tool


# Global state for scheduling
class SchedulingState:
    """Global state shared by all tools."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.scheduled_entries = []  # list of dicts
        self.teacher_schedule = defaultdict(set)  # (day, hour) -> set of teachers
        self.room_schedule = defaultdict(bool)  # (day, hour, room) -> occupied
        self.student_conflicts = defaultdict(set)  # (course, batch) -> set of conflicting (course, batch)
        self.slot_schedule = defaultdict(set)  # (day, hour) -> set of (course, batch)
        self.courses = {}  # code -> course info dict
        self.course_progress = defaultdict(lambda: {"theory": 0, "lab": 0})
    
    def load_data(self, data_dir: str):
        """Load course and student data."""
        data_path = Path(data_dir)
        
        # Load courses and teachers from course_batch_teachers.csv
        with open(data_path / "course_batch_teachers.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row["CourseCode"]
                course_type = row.get("CourseType", "")
                batch = row["BatchID"]
                teacher = row["TeacherName"]
                
                # Parse hours from CourseType like "Theory 4hr" or "Lab 2hr"
                theory_hours = 0
                lab_hours = 0
                if "Theory" in course_type:
                    try:
                        theory_hours = int(course_type.split()[1].replace("hr", ""))
                    except:
                        theory_hours = 4
                if "Lab" in course_type:
                    try:
                        lab_hours = int(course_type.split()[-1].replace("hr", ""))
                    except:
                        lab_hours = 2
                
                if code not in self.courses:
                    self.courses[code] = {
                        "code": code,
                        "theory_hours": theory_hours,
                        "lab_hours": lab_hours,
                        "batches": [],
                        "teachers": {}
                    }
                
                if batch not in self.courses[code]["batches"]:
                    self.courses[code]["batches"].append(batch)
                self.courses[code]["teachers"][batch] = teacher
        
        # Build student conflict matrix
        with open(data_path / "student_allocations_aggregated.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                courses = row['Allocated Courses'].replace('"', '').split(', ')
                batches = row['Batches'].replace('"', '').split(', ')
                
                course_batches = [(c.strip(), b.strip()) for c, b in zip(courses, batches)]
                for i, cb1 in enumerate(course_batches):
                    for cb2 in course_batches[i+1:]:
                        self.student_conflicts[cb1].add(cb2)
                        self.student_conflicts[cb2].add(cb1)
        
        return f"Loaded {len(self.courses)} courses with student conflicts"


# Global state instance
state = SchedulingState()


@tool
def load_scheduling_data(data_dir: str) -> str:
    """
    Load all scheduling data from CSV files.
    Args:
        data_dir: Path to directory containing CSV files
    Returns:
        Summary of loaded data
    """
    state.reset()
    result = state.load_data(data_dir)
    return result


@tool
def get_courses_to_schedule() -> str:
    """
    Get list of all courses that need to be scheduled.
    Returns:
        JSON string of courses with their requirements
    """
    import json
    courses_list = []
    for code, info in state.courses.items():
        for batch in info.get("batches", ["B1"]):
            teacher = info.get("teachers", {}).get(batch, "TBA")
            courses_list.append({
                "course": code,
                "batch": batch,
                "teacher": teacher,
                "theory_hours": info["theory_hours"],
                "lab_hours": info["lab_hours"]
            })
    return json.dumps(courses_list[:50], indent=2)  # Return first 50


@tool
def check_slot_available(day: str, hour: int, teacher: str, room: str, course: str, batch: str) -> str:
    """
    Check if a time slot is available for scheduling.
    Args:
        day: Day of week (Monday-Friday)
        hour: Hour (10-17)
        teacher: Teacher name
        room: Room ID (R1-R21 for theory, LAB1-LAB7 for labs)
        course: Course code
        batch: Batch ID
    Returns:
        Availability status and reason
    """
    # Check teacher
    if teacher in state.teacher_schedule[(day, hour)]:
        return f"NOT AVAILABLE: {teacher} already has a class at {day} {hour}:00"
    
    # Check room
    if state.room_schedule[(day, hour, room)]:
        return f"NOT AVAILABLE: Room {room} already occupied at {day} {hour}:00"
    
    # Check student conflicts
    cb = (course, batch)
    for scheduled_cb in state.slot_schedule[(day, hour)]:
        if scheduled_cb in state.student_conflicts.get(cb, set()):
            return f"NOT AVAILABLE: Students in {course}-{batch} have class {scheduled_cb[0]}-{scheduled_cb[1]} at this time"
    
    return f"AVAILABLE: Slot {day} {hour}:00 in {room} is free for {teacher}"


@tool
def assign_slot(day: str, hour: int, teacher: str, room: str, course: str, batch: str, session_type: str) -> str:
    """
    Assign a course to a specific time slot.
    Args:
        day: Day of week (Monday-Friday)
        hour: Hour (10-17)
        teacher: Teacher name
        room: Room ID
        course: Course code
        batch: Batch ID
        session_type: "theory" or "lab"
    Returns:
        Success or failure message
    """
    # Validate
    if hour < 10 or hour > 17:
        return "ERROR: Hour must be between 10 and 17"
    
    if day not in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        return f"ERROR: Invalid day {day}"
    
    # Check availability first
    if teacher in state.teacher_schedule[(day, hour)]:
        return f"FAILED: {teacher} already busy at {day} {hour}:00"
    
    if state.room_schedule[(day, hour, room)]:
        return f"FAILED: Room {room} already occupied at {day} {hour}:00"
    
    # Check student conflicts
    cb = (course, batch)
    for scheduled_cb in state.slot_schedule[(day, hour)]:
        if scheduled_cb in state.student_conflicts.get(cb, set()):
            return f"FAILED: Student conflict with {scheduled_cb[0]}-{scheduled_cb[1]}"
    
    # Assign
    entry = {
        "day": day,
        "hour": hour,
        "teacher": teacher,
        "room": room,
        "course": course,
        "batch": batch,
        "type": session_type
    }
    state.scheduled_entries.append(entry)
    state.teacher_schedule[(day, hour)].add(teacher)
    state.room_schedule[(day, hour, room)] = True
    state.slot_schedule[(day, hour)].add(cb)
    
    # Update progress
    if session_type == "lab":
        state.course_progress[(course, batch)]["lab"] += 1
    else:
        state.course_progress[(course, batch)]["theory"] += 1
    
    return f"SUCCESS: Assigned {course}-{batch} to {day} {hour}:00 in {room} with {teacher}"


@tool
def get_schedule_status() -> str:
    """
    Get current scheduling progress.
    Returns:
        Summary of scheduled entries and incomplete courses
    """
    total_scheduled = len(state.scheduled_entries)
    
    # Count incomplete courses
    incomplete = []
    for code, info in state.courses.items():
        for batch in info.get("batches", ["B1"]):
            progress = state.course_progress[(code, batch)]
            theory_needed = info["theory_hours"] - progress["theory"]
            lab_needed = info["lab_hours"] - progress["lab"]
            if theory_needed > 0 or lab_needed > 0:
                incomplete.append(f"{code}-{batch}: needs {theory_needed}T/{lab_needed}L")
    
    return f"""
SCHEDULE STATUS:
- Total scheduled: {total_scheduled} sessions
- Incomplete courses: {len(incomplete)}
- Sample incomplete: {incomplete[:10]}
"""


@tool
def verify_schedule() -> str:
    """
    Verify the current schedule for conflicts.
    Returns:
        Verification report with any conflicts found
    """
    room_conflicts = 0
    teacher_conflicts = 0
    student_conflicts = 0
    
    # Check room and teacher conflicts (should be 0 if using tools correctly)
    room_check = defaultdict(list)
    teacher_check = defaultdict(list)
    
    for entry in state.scheduled_entries:
        key = (entry["day"], entry["hour"], entry["room"])
        room_check[key].append(entry["course"])
        
        t_key = (entry["day"], entry["hour"], entry["teacher"])
        teacher_check[t_key].append(entry["course"])
    
    for key, courses in room_check.items():
        if len(courses) > 1:
            room_conflicts += 1
    
    for key, courses in teacher_check.items():
        if len(courses) > 1:
            teacher_conflicts += 1
    
    return f"""
VERIFICATION REPORT:
- Room conflicts: {room_conflicts}
- Teacher conflicts: {teacher_conflicts}
- Student conflicts: Checked during assignment (should be 0)
- Total entries: {len(state.scheduled_entries)}
"""


@tool
def save_schedule(output_path: str) -> str:
    """
    Save the schedule to a CSV file.
    Args:
        output_path: Path to save the CSV file
    Returns:
        Success message
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Day", "Hour", "Course", "Batch", "Teacher", "Room", "Type"])
        
        sorted_entries = sorted(
            state.scheduled_entries,
            key=lambda e: (
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(e["day"]),
                e["hour"]
            )
        )
        
        for entry in sorted_entries:
            writer.writerow([
                entry["day"],
                f"{entry['hour']}:00-{entry['hour']+1}:00",
                entry["course"],
                entry["batch"],
                entry["teacher"],
                entry["room"],
                entry["type"]
            ])
    
    return f"SUCCESS: Saved {len(state.scheduled_entries)} entries to {output_path}"


@tool
def get_available_slots_for_course(course: str, batch: str, session_type: str) -> str:
    """
    Find available slots for a specific course-batch.
    Args:
        course: Course code
        batch: Batch ID
        session_type: "theory" or "lab"
    Returns:
        List of available (day, hour, room) combinations
    """
    if course not in state.courses:
        return f"ERROR: Course {course} not found"
    
    teacher = state.courses[course].get("teachers", {}).get(batch, "TBA")
    rooms = [f"R{i}" for i in range(1, 22)] if session_type == "theory" else [f"LAB{i}" for i in range(1, 8)]
    
    available = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    for day in days:
        for hour in range(10, 18):
            # Check teacher
            if teacher in state.teacher_schedule[(day, hour)]:
                continue
            
            # Check student conflicts
            cb = (course, batch)
            has_student_conflict = False
            for scheduled_cb in state.slot_schedule[(day, hour)]:
                if scheduled_cb in state.student_conflicts.get(cb, set()):
                    has_student_conflict = True
                    break
            
            if has_student_conflict:
                continue
            
            # Find available room
            for room in rooms:
                if not state.room_schedule[(day, hour, room)]:
                    available.append(f"{day} {hour}:00 in {room}")
                    break
    
    return f"Available slots for {course}-{batch} ({teacher}):\n" + "\n".join(available[:10])
