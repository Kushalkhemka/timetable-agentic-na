"""
Data loader for CSV files.
"""
import pandas as pd
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_models import (
    Course, Teacher, Student, CourseType, SchedulingConfig
)


class DataLoader:
    """Loads and parses CSV data into structured models."""
    
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._courses: dict[str, Course] = {}
        self._teachers: dict[str, Teacher] = {}
        self._students: dict[str, Student] = {}
        self._loaded = False
    
    def load_all(self) -> None:
        """Load all CSV files and build data structures."""
        self._load_course_batches()
        self._load_course_batch_teachers()
        self._load_student_allocations()
        self._loaded = True
    
    def _load_course_batches(self) -> None:
        """Load course_batches.csv"""
        df = pd.read_csv(self.data_dir / "course_batches.csv")
        
        for _, row in df.iterrows():
            code = row["CourseCode"]
            if pd.isna(code) or not code:
                continue
                
            # Parse batch sizes
            batch_mode = str(row["BatchMode"])
            if batch_mode.startswith('"'):
                batch_mode = batch_mode.strip('"')
            batch_sizes = [int(x.strip()) for x in batch_mode.split(",") if x.strip()]
            
            # Parse batch IDs
            batches_str = str(row["Batches"])
            if batches_str.startswith('"'):
                batches_str = batches_str.strip('"')
            batches = [x.strip() for x in batches_str.split(",") if x.strip()]
            
            self._courses[code] = Course(
                code=code,
                course_type=CourseType.THEORY_4HR,  # Will be updated in next load
                total_batches=int(row["TotalBatches"]),
                batches=batches,
                batch_sizes=batch_sizes,
                teacher_assignments={}
            )
    
    def _load_course_batch_teachers(self) -> None:
        """Load course_batch_teachers.csv and update courses with types and teachers."""
        df = pd.read_csv(self.data_dir / "course_batch_teachers.csv")
        
        for _, row in df.iterrows():
            code = row["CourseCode"]
            if pd.isna(code) or not code:
                continue
            
            batch_id = row["BatchID"]
            teacher_name = row["TeacherName"]
            course_type_str = row["CourseType"]
            
            # Update course type - support both formats:
            # Old: "Theory 4hr", "Theory 3hr + Lab 2hr"
            # New: "4", "3+2"
            if code in self._courses:
                course_type_str = str(course_type_str).strip()
                if "Lab" in course_type_str or "+2" in course_type_str or course_type_str == "3+2":
                    self._courses[code].course_type = CourseType.THEORY_3HR_LAB_2HR
                else:
                    self._courses[code].course_type = CourseType.THEORY_4HR
                    
                # Add teacher assignment
                self._courses[code].teacher_assignments[batch_id] = teacher_name
            
            # Track teachers
            if teacher_name not in self._teachers:
                self._teachers[teacher_name] = Teacher(name=teacher_name, courses=[])
            if code not in self._teachers[teacher_name].courses:
                self._teachers[teacher_name].courses.append(code)
    
    def _load_student_allocations(self) -> None:
        """Load student_allocations_aggregated.csv"""
        df = pd.read_csv(self.data_dir / "student_allocations_aggregated.csv")
        
        for _, row in df.iterrows():
            roll_no = row["roll_no"]
            if pd.isna(roll_no) or not roll_no:
                continue
                
            name = row["name"] if pd.notna(row["name"]) else ""
            
            # Parse courses
            courses_str = str(row["Allocated Courses"])
            if courses_str.startswith('"'):
                courses_str = courses_str.strip('"')
            courses = [x.strip() for x in courses_str.split(",") if x.strip()]
            
            # Parse batches
            batches_str = str(row["Batches"])
            if batches_str.startswith('"'):
                batches_str = batches_str.strip('"')
            batches = [x.strip() for x in batches_str.split(",") if x.strip()]
            
            # Create batch assignments mapping
            batch_assignments = {}
            for i, course in enumerate(courses):
                if i < len(batches):
                    batch_assignments[course] = batches[i]
            
            self._students[roll_no] = Student(
                roll_no=roll_no,
                name=name,
                allocated_courses=courses,
                batch_assignments=batch_assignments
            )
    
    @property
    def courses(self) -> dict[str, Course]:
        if not self._loaded:
            self.load_all()
        return self._courses
    
    @property
    def teachers(self) -> dict[str, Teacher]:
        if not self._loaded:
            self.load_all()
        return self._teachers
    
    @property
    def students(self) -> dict[str, Student]:
        if not self._loaded:
            self.load_all()
        return self._students
    
    def get_course_batch_sessions(self) -> list[dict]:
        """
        Get all sessions that need to be scheduled.
        Returns list of dicts with course, batch, teacher, session_type, hours needed.
        """
        sessions = []
        
        for code, course in self.courses.items():
            for batch_id in course.batches:
                teacher = course.teacher_assignments.get(batch_id, "TBA")
                batch_idx = course.batches.index(batch_id)
                student_count = course.batch_sizes[batch_idx] if batch_idx < len(course.batch_sizes) else 0
                
                # Theory sessions
                for hour in range(course.theory_hours):
                    sessions.append({
                        "course_code": code,
                        "batch_id": batch_id,
                        "teacher_name": teacher,
                        "session_type": "Theory",
                        "session_number": hour + 1,
                        "student_count": student_count
                    })
                
                # Lab sessions
                for hour in range(course.lab_hours):
                    sessions.append({
                        "course_code": code,
                        "batch_id": batch_id,
                        "teacher_name": teacher,
                        "session_type": "Lab",
                        "session_number": hour + 1,
                        "student_count": student_count
                    })
        
        return sessions
    
    def get_stats(self) -> dict:
        """Get statistics about the loaded data."""
        if not self._loaded:
            self.load_all()
            
        theory_courses = sum(1 for c in self._courses.values() if c.course_type == CourseType.THEORY_4HR)
        lab_courses = sum(1 for c in self._courses.values() if c.course_type == CourseType.THEORY_3HR_LAB_2HR)
        
        total_sessions = len(self.get_course_batch_sessions())
        
        return {
            "total_courses": len(self._courses),
            "theory_only_courses": theory_courses,
            "theory_plus_lab_courses": lab_courses,
            "total_teachers": len(self._teachers),
            "total_students": len(self._students),
            "total_sessions_to_schedule": total_sessions
        }


if __name__ == "__main__":
    # Test loading
    loader = DataLoader(Path(__file__).parent.parent)
    loader.load_all()
    
    print("=== Data Loading Test ===")
    stats = loader.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Sample Course ===")
    sample_course = list(loader.courses.values())[0]
    print(f"Code: {sample_course.code}")
    print(f"Type: {sample_course.course_type}")
    print(f"Batches: {sample_course.batches}")
    print(f"Teachers: {sample_course.teacher_assignments}")
