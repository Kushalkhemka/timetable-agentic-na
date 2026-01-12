"""
CrewAI Tasks for Timetable Scheduling.
Defines the tasks that agents perform in sequence.
"""
from crewai import Task


def create_analysis_task(constraint_agent, data_dir: str):
    """Task to analyze scheduling data and constraints."""
    return Task(
        description=f"""
        CRITICAL: You MUST load data before anything else.
        
        STEP 1 (REQUIRED): Call load_scheduling_data with data_dir="{data_dir}"
        This loads all course, teacher, and student data.
        
        STEP 2: Call get_courses_to_schedule to see all courses that need scheduling.
        
        STEP 3: Analyze and report:
        - Total courses and batches to schedule
        - Total hours needed
        - Student conflict density
        
        DO NOT SKIP STEP 1. The load_scheduling_data tool MUST be called first.
        """,
        expected_output="Confirmation that data was loaded, plus list of courses to schedule",
        agent=constraint_agent
    )


def create_scheduling_task(planner_agent, data_dir: str):
    """Task to schedule all courses - one batch at a time."""
    return Task(
        description=f"""
        YOU MUST SCHEDULE ALL COURSES. DO NOT STOP UNTIL COMPLETE.
        
        PROCESS ONE COURSE-BATCH AT A TIME:
        
        For EACH course-batch (e.g., CO403-B1):
        1. Call get_available_slots_for_course to see free slots
        2. For each theory hour needed:
           - Call assign_slot with day, hour, teacher, room, type="theory"
        3. For each lab hour needed (use LAB rooms, consecutive 2hrs):
           - Call assign_slot with lab room (LAB1-LAB7), type="lab"
        4. Check get_schedule_status periodically
        
        ROOMS:
        - Theory: R1 through R21
        - Labs: LAB1 through LAB7
        
        HOURS: 10, 11, 12, 13, 14, 15, 16, 17
        DAYS: Monday, Tuesday, Wednesday, Thursday, Friday
        
        START NOW. Schedule CO403-B1 first, then CO403-B2, etc.
        
        CRITICAL: You have 1 MILLION token context. Continue scheduling 
        until get_schedule_status shows 0 incomplete courses or you've 
        tried all 235 course-batches.
        
        DO NOT give a final answer until you've scheduled at least 50 courses.
        """,
        expected_output="Summary showing: X courses scheduled, Y total sessions, Z incomplete",
        agent=planner_agent
    )


def create_verification_task(verification_agent):
    """Task to verify the schedule."""
    return Task(
        description="""
        Verify the generated schedule for any conflicts.
        
        Use verify_schedule to check:
        1. Room conflicts (same room, same time, different courses)
        2. Teacher conflicts (same teacher, same time, different rooms)
        3. Student conflicts (courses sharing students at same time)
        
        Use get_schedule_status to check:
        - Total sessions scheduled
        - Which courses are incomplete
        
        Report:
        - Total conflicts found (should be 0)
        - Coverage percentage
        - List of incomplete courses if any
        """,
        expected_output="Verification report with conflict counts and coverage percentage",
        agent=verification_agent
    )


def create_fixing_task(fixer_agent, output_path: str):
    """Task to fix any remaining issues and save."""
    return Task(
        description=f"""
        Complete any remaining scheduling and save the final timetable.
        
        1. Check get_schedule_status for incomplete courses
        2. For each incomplete course:
           - Use get_available_slots_for_course to find free slots
           - Use assign_slot to schedule remaining hours
        3. Once complete, use save_schedule to save to {output_path}
        
        Continue until:
        - All courses are fully scheduled, OR
        - No more available slots remain
        
        Save the final schedule even if incomplete.
        """,
        expected_output="Final schedule saved with summary of coverage achieved",
        agent=fixer_agent
    )
