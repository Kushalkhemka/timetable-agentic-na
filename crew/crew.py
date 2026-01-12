"""
Main Crew orchestration for Timetable Scheduling.
Brings together agents and tasks into a collaborative workflow.
"""
from crewai import Crew, Process
from .agents import get_all_agents
from .tasks import (
    create_analysis_task,
    create_scheduling_task,
    create_verification_task,
    create_fixing_task
)


class TimetableCrew:
    """
    CrewAI-based multi-agent timetable scheduling system.
    
    Agents:
    - Constraint Agent: Analyzes data and constraints
    - Planner Agent: Assigns courses to slots
    - Verification Agent: Checks for conflicts
    - Fixer Agent: Resolves issues and saves output
    """
    
    def __init__(self, data_dir: str = ".", output_dir: str = "./output"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.agents = get_all_agents()
    
    def run(self, verbose: bool = True) -> str:
        """
        Run the multi-agent scheduling workflow.
        
        Returns:
            Final result message
        """
        print("="*60)
        print("🤖 CrewAI Multi-Agent Timetable Scheduling")
        print("="*60)
        
        # Create tasks
        analysis_task = create_analysis_task(
            self.agents["constraint"],
            self.data_dir
        )
        
        scheduling_task = create_scheduling_task(
            self.agents["planner"],
            self.data_dir
        )
        
        verification_task = create_verification_task(
            self.agents["verifier"]
        )
        
        fixing_task = create_fixing_task(
            self.agents["fixer"],
            f"{self.output_dir}/timetable.csv"
        )
        
        # Create crew with sequential process
        crew = Crew(
            agents=[
                self.agents["constraint"],
                self.agents["planner"],
                self.agents["verifier"],
                self.agents["fixer"]
            ],
            tasks=[
                analysis_task,
                scheduling_task,
                verification_task,
                fixing_task
            ],
            process=Process.sequential,
            verbose=verbose
        )
        
        # Run the crew
        print("\n🚀 Starting multi-agent scheduling...\n")
        result = crew.kickoff()
        
        print("\n" + "="*60)
        print("✅ Crew Scheduling Complete!")
        print("="*60)
        
        return str(result)


def run_timetable_crew(data_dir: str = ".", output_dir: str = "./output", verbose: bool = True):
    """Convenience function to run the timetable crew."""
    crew = TimetableCrew(data_dir, output_dir)
    return crew.run(verbose=verbose)
