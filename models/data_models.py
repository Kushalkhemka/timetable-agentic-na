"""
Data models for the multi-agent timetable scheduling system.
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Day(str, Enum):
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"


class CourseType(str, Enum):
    THEORY_4HR = "Theory 4hr"
    THEORY_3HR_LAB_2HR = "Theory 3hr + Lab 2hr"


class SessionType(str, Enum):
    THEORY = "Theory"
    LAB = "Lab"


class TimeSlot(BaseModel):
    """Represents a 1-hour time slot"""
    day: Day
    hour: int = Field(ge=10, le=17, description="Hour in 24h format (10-17 for 10am-6pm)")
    
    def __hash__(self):
        return hash((self.day.value, self.hour))
    
    def __eq__(self, other):
        return self.day == other.day and self.hour == other.hour
    
    @property
    def display(self) -> str:
        return f"{self.day.value} {self.hour}:00-{self.hour+1}:00"


class Room(BaseModel):
    """Represents a classroom/lab room"""
    room_id: str
    capacity: int = Field(default=90, ge=1)
    
    def __hash__(self):
        return hash(self.room_id)


class Teacher(BaseModel):
    """Represents a faculty member"""
    name: str
    courses: list[str] = Field(default_factory=list, description="Course codes taught")
    
    def __hash__(self):
        return hash(self.name)


class Course(BaseModel):
    """Represents a course with its batches"""
    code: str
    course_type: CourseType
    total_batches: int
    batches: list[str] = Field(description="Batch IDs like B1, B2...")
    batch_sizes: list[int] = Field(description="Number of students per batch")
    teacher_assignments: dict[str, str] = Field(
        default_factory=dict, 
        description="Mapping of batch_id -> teacher_name"
    )
    
    @property
    def theory_hours(self) -> int:
        return 4 if self.course_type == CourseType.THEORY_4HR else 3
    
    @property
    def lab_hours(self) -> int:
        return 0 if self.course_type == CourseType.THEORY_4HR else 2
    
    @property
    def total_hours(self) -> int:
        return self.theory_hours + self.lab_hours


class Student(BaseModel):
    """Represents a student with course allocations"""
    roll_no: str
    name: str
    allocated_courses: list[str] = Field(description="Course codes")
    batch_assignments: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of course_code -> batch_id"
    )


class ScheduleEntry(BaseModel):
    """A single entry in the timetable"""
    course_code: str
    batch_id: str
    teacher_name: str
    room_id: str
    time_slot: TimeSlot
    session_type: SessionType
    student_count: int
    
    def __hash__(self):
        return hash((self.course_code, self.batch_id, self.time_slot.day.value, self.time_slot.hour))


class Constraint(BaseModel):
    """Represents a scheduling constraint"""
    constraint_id: str
    constraint_type: str = Field(description="hard or soft")
    description: str
    entities_involved: list[str] = Field(default_factory=list)


class TimetableProposal(BaseModel):
    """A proposed timetable solution"""
    proposal_id: str
    entries: list[ScheduleEntry] = Field(default_factory=list)
    algorithm_used: str = Field(default="greedy")
    generation_time_ms: float = Field(default=0.0)
    
    @property
    def total_scheduled_hours(self) -> int:
        return len(self.entries)


class VerificationResult(BaseModel):
    """Result from the verification agent"""
    proposal_id: str
    is_valid: bool
    score: float = Field(ge=0.0, le=1.0, description="Quality score 0-1")
    conflicts: list[dict] = Field(default_factory=list)
    feedback: str = Field(default="")
    suggestions: list[str] = Field(default_factory=list)


class SchedulingConfig(BaseModel):
    """Configuration for the scheduling problem"""
    days: list[Day] = Field(
        default=[Day.MONDAY, Day.TUESDAY, Day.WEDNESDAY, Day.THURSDAY, Day.FRIDAY]
    )
    start_hour: int = Field(default=10, ge=0, le=23)
    end_hour: int = Field(default=18, ge=1, le=24)
    num_rooms: int = Field(default=10, ge=1)
    room_capacity: int = Field(default=90, ge=1)
    
    @property
    def slots_per_day(self) -> int:
        return self.end_hour - self.start_hour
    
    @property
    def total_slots(self) -> int:
        return len(self.days) * self.slots_per_day
    
    def generate_all_time_slots(self) -> list[TimeSlot]:
        """Generate all available time slots"""
        slots = []
        for day in self.days:
            for hour in range(self.start_hour, self.end_hour):
                slots.append(TimeSlot(day=day, hour=hour))
        return slots
    
    def generate_all_rooms(self) -> list[Room]:
        """Generate all available rooms"""
        return [Room(room_id=f"R{i+1}", capacity=self.room_capacity) for i in range(self.num_rooms)]
