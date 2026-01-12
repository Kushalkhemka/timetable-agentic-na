"""
LangChain Multi-Agent Timetable Scheduler.
Uses LangChain with direct Gemini function calling for reliable scheduling.
"""
import os
import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

load_dotenv()


class SchedulingState:
    """Global scheduling state."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.entries = []
        self.teacher_schedule = defaultdict(set)
        self.room_schedule = defaultdict(bool)
        self.student_conflicts = defaultdict(set)
        self.slot_schedule = defaultdict(set)
        self.courses = {}
        self.progress = defaultdict(lambda: {"theory": 0, "lab": 0})
    
    def load_data(self, data_dir: str):
        """Load all scheduling data."""
        data_path = Path(data_dir)
        
        # Load courses from course_batch_teachers.csv
        with open(data_path / "course_batch_teachers.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row["CourseCode"]
                course_type = row.get("CourseType", "")
                batch = row["BatchID"]
                teacher = row["TeacherName"]
                
                theory_hours = 4 if "Theory" in course_type else 0
                lab_hours = 2 if "Lab" in course_type else 0
                
                if code not in self.courses:
                    self.courses[code] = {
                        "theory_hours": theory_hours,
                        "lab_hours": lab_hours,
                        "batches": [],
                        "teachers": {}
                    }
                
                if batch not in self.courses[code]["batches"]:
                    self.courses[code]["batches"].append(batch)
                self.courses[code]["teachers"][batch] = teacher
        
        # Load student conflicts
        with open(data_path / "student_allocations_aggregated.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                courses = row['Allocated Courses'].replace('"', '').split(', ')
                batches = row['Batches'].replace('"', '').split(', ')
                
                cbs = [(c.strip(), b.strip()) for c, b in zip(courses, batches)]
                for i, cb1 in enumerate(cbs):
                    for cb2 in cbs[i+1:]:
                        self.student_conflicts[cb1].add(cb2)
                        self.student_conflicts[cb2].add(cb1)
        
        return f"Loaded {len(self.courses)} courses"


# Global state
state = SchedulingState()


def load_data(data_dir: str) -> str:
    """Load scheduling data from CSV files."""
    state.reset()
    return state.load_data(data_dir)


def get_courses() -> str:
    """Get list of courses to schedule."""
    courses = []
    for code, info in list(state.courses.items())[:30]:
        for batch in info["batches"]:
            teacher = info["teachers"].get(batch, "TBA")
            courses.append(f"{code}-{batch} ({teacher}): {info['theory_hours']}T/{info['lab_hours']}L")
    return "\n".join(courses)


def assign_slot(course: str, batch: str, day: str, hour: int, room: str, session_type: str) -> str:
    """Assign a course to a slot. Returns success/failure."""
    teacher = state.courses.get(course, {}).get("teachers", {}).get(batch, "TBA")
    
    # Check teacher
    if teacher in state.teacher_schedule[(day, hour)]:
        return f"FAILED: {teacher} busy at {day} {hour}"
    
    # Check room
    if state.room_schedule[(day, hour, room)]:
        return f"FAILED: {room} occupied at {day} {hour}"
    
    # Check student conflicts
    cb = (course, batch)
    for scheduled in state.slot_schedule[(day, hour)]:
        if scheduled in state.student_conflicts.get(cb, set()):
            return f"FAILED: Student conflict with {scheduled}"
    
    # Assign
    state.entries.append({
        "day": day, "hour": hour, "course": course, "batch": batch,
        "teacher": teacher, "room": room, "type": session_type
    })
    state.teacher_schedule[(day, hour)].add(teacher)
    state.room_schedule[(day, hour, room)] = True
    state.slot_schedule[(day, hour)].add(cb)
    state.progress[(course, batch)][session_type] += 1
    
    return f"SUCCESS: {course}-{batch} assigned to {day} {hour}:00 in {room}"


def get_status() -> str:
    """Get scheduling status."""
    incomplete = []
    for code, info in state.courses.items():
        for batch in info["batches"]:
            p = state.progress[(code, batch)]
            t_need = info["theory_hours"] - p["theory"]
            l_need = info["lab_hours"] - p["lab"]
            if t_need > 0 or l_need > 0:
                incomplete.append(f"{code}-{batch}: {t_need}T/{l_need}L")
    
    return f"Scheduled: {len(state.entries)} | Incomplete: {len(incomplete)}"


def save_schedule(output_path: str) -> str:
    """Save schedule to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Day", "Hour", "Course", "Batch", "Teacher", "Room", "Type"])
        for e in sorted(state.entries, key=lambda x: (
            ["Monday","Tuesday","Wednesday","Thursday","Friday"].index(x["day"]), x["hour"]
        )):
            w.writerow([e["day"], f"{e['hour']}:00", e["course"], e["batch"], 
                       e["teacher"], e["room"], e["type"]])
    return f"Saved {len(state.entries)} entries to {output_path}"


def parse_assign_slot_input(x: str) -> str:
    """Parse assign_slot input - handles 'CO403, B1, Monday, 10, R1, theory' format."""
    # Remove any parentheses
    x = x.strip().strip("()").strip()
    
    # Split by comma
    parts = [p.strip().strip("'\"") for p in x.split(",")]
    
    if len(parts) != 6:
        return f"FAILED: Expected 6 parts (course, batch, day, hour, room, type), got {len(parts)}: {parts}"
    
    course, batch, day, hour_str, room, session_type = parts
    
    try:
        hour = int(hour_str)
    except:
        return f"FAILED: Hour must be integer, got {hour_str}"
    
    return assign_slot(course, batch, day, hour, room, session_type)


# Create LangChain tools
tools = [
    Tool(name="load_data", func=lambda x: load_data("."), 
         description="Load scheduling data. Call first."),
    Tool(name="get_courses", func=lambda x: get_courses(),
         description="Get list of courses to schedule"),
    Tool(name="assign_slot", 
         func=parse_assign_slot_input,
         description="assign_slot(course, batch, day, hour, room, type). "
                    "Input format: 'CO403, B1, Monday, 10, R1, theory'. "
                    "Days: Monday-Friday. Hours: 10-17. "
                    "Rooms: R1-R21 (theory), LAB1-LAB7 (labs)."),
    Tool(name="get_status", func=lambda x: get_status(),
         description="Get current scheduling status"),
    Tool(name="save_schedule", func=lambda x: save_schedule("./output/timetable.csv"),
         description="Save the schedule to CSV"),
]


def create_scheduler_agent():
    """Create LangChain agent for scheduling."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    prompt = PromptTemplate.from_template("""You are a timetable scheduler. Schedule courses one by one.

Tools: {tools}
Tool Names: {tool_names}

WORKFLOW:
1. load_data first
2. get_courses to see what needs scheduling
3. For each course-batch, call assign_slot with appropriate parameters
4. Check get_status periodically
5. When done or stuck, call save_schedule

IMPORTANT:
- Theory uses R1-R21
- Labs use LAB1-LAB7
- Hours are 10, 11, 12, 13, 14, 15, 16, 17
- Days are Monday, Tuesday, Wednesday, Thursday, Friday

Format:
Thought: what to do next
Action: tool_name
Action Input: input for the tool
Observation: result
... (repeat)
Thought: I'm done
Final Answer: summary

Begin!

{agent_scratchpad}""")
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=100)


def run_langchain_scheduler():
    """Run the LangChain scheduler."""
    print("="*60)
    print("🤖 LangChain Multi-Agent Timetable Scheduler")
    print("="*60)
    
    executor = create_scheduler_agent()
    
    result = executor.invoke({
        "input": """Schedule ALL courses. 
        Start with load_data, then get_courses.
        Then assign_slot for each course-batch, one at a time.
        Schedule at least 20 courses before saving.
        Call save_schedule when done."""
    })
    
    print("\n" + "="*60)
    print("✅ Complete!")
    print(f"Result: {result['output']}")
    print("="*60)
    
    return result


if __name__ == "__main__":
    run_langchain_scheduler()
