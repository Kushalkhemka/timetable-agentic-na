"""
CrewAI Agents for Timetable Scheduling.
5 specialized agents that collaborate to create a conflict-free timetable.
"""
import os
from crewai import Agent, LLM
from .tools import (
    load_scheduling_data,
    get_courses_to_schedule,
    check_slot_available,
    assign_slot,
    get_schedule_status,
    verify_schedule,
    save_schedule,
    get_available_slots_for_course
)


def get_llm():
    """Get the Gemini LLM for agents."""
    return LLM(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )


def create_manager_agent():
    """Manager Agent - Coordinates the entire scheduling workflow."""
    return Agent(
        role="Scheduling Manager",
        goal="Coordinate the multi-agent team to create a complete, conflict-free timetable for all courses",
        backstory="""You are an expert academic administrator who manages the timetable 
        scheduling process. You delegate tasks to specialized agents and ensure 
        the final schedule meets all constraints: no room conflicts, no teacher 
        conflicts, and no student conflicts.""",
        llm=get_llm(),
        verbose=True,
        allow_delegation=True
    )


def create_constraint_agent():
    """Constraint Agent - Analyzes data and identifies constraints."""
    return Agent(
        role="Constraint Analyst",
        goal="Analyze scheduling data and identify all hard and soft constraints",
        backstory="""You are a data analyst specializing in constraint identification.
        You analyze course data, teacher assignments, and student enrollments to 
        identify which courses share students (can't be scheduled together),
        which teachers are overloaded, and what the capacity constraints are.""",
        llm=get_llm(),
        tools=[load_scheduling_data, get_courses_to_schedule],
        verbose=True
    )


def create_planner_agent():
    """Planner Agent - Assigns courses to time slots."""
    return Agent(
        role="Schedule Planner",
        goal="Assign all courses to time slots while avoiding all conflicts",
        backstory="""You are an expert scheduler who knows how to fit courses into 
        available slots. You always check slot availability before assigning.
        You prioritize:
        1. Lab sessions first (they need consecutive 2-hour blocks)
        2. Courses with most student conflicts (hardest to schedule)
        3. Theory sessions to fill remaining slots
        
        You use the check_slot_available tool before every assignment to ensure
        no teacher, room, or student conflicts occur.""",
        llm=get_llm(),
        tools=[
            check_slot_available,
            assign_slot,
            get_available_slots_for_course,
            get_schedule_status
        ],
        verbose=True
    )


def create_verification_agent():
    """Verification Agent - Checks schedule for conflicts."""
    return Agent(
        role="Schedule Verifier",
        goal="Verify the schedule is conflict-free and complete",
        backstory="""You are a quality assurance specialist who meticulously checks
        schedules for any issues. You verify:
        - No room is double-booked
        - No teacher teaches two classes at once
        - No student has overlapping courses
        - All required hours are scheduled for each course""",
        llm=get_llm(),
        tools=[verify_schedule, get_schedule_status],
        verbose=True
    )


def create_fixer_agent():
    """Fixer Agent - Resolves conflicts and fills gaps."""
    return Agent(
        role="Schedule Fixer",
        goal="Resolve any conflicts and complete any missing course assignments",
        backstory="""You are a problem-solver who specializes in fixing scheduling
        conflicts. When the verification reveals issues, you:
        - Identify alternative slots for conflicting courses
        - Reassign teachers if needed
        - Ensure all courses get their required hours""",
        llm=get_llm(),
        tools=[
            check_slot_available,
            assign_slot,
            get_available_slots_for_course,
            get_schedule_status,
            save_schedule
        ],
        verbose=True
    )


def get_all_agents():
    """Create and return all agents."""
    return {
        "manager": create_manager_agent(),
        "constraint": create_constraint_agent(),
        "planner": create_planner_agent(),
        "verifier": create_verification_agent(),
        "fixer": create_fixer_agent()
    }
