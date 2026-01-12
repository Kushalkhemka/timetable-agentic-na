"""
Orchestrator - Coordinates the multi-agent timetable scheduling workflow with memory.
Implements iterative refinement following PlanGEN framework.
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

from agents import ConstraintAgent, PlannerAgent, LLMPlannerAgent, ToolBasedPlannerAgent, VerificationAgent, SelectionAgent, AgentMemory
from models import Course, Teacher, SchedulingConfig, TimetableProposal, VerificationResult
from utils import DataLoader


class SchedulingOrchestrator:
    """
    Orchestrates the PlanGEN-inspired multi-agent workflow with memory:
    1. Constraint Agent extracts constraints
    2. Selection Agent picks algorithm based on complexity
    3. Planner Agent generates proposals using LLM
    4. Verification Agent validates and scores
    5. Memory records trajectories and learnings
    6. Iterate with refinement until valid or max iterations
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        config: Optional[SchedulingConfig] = None,
        api_key: Optional[str] = None,
        max_iterations: int = 5,
        use_full_llm: bool = True  # Use true LLM-driven scheduling
    ):
        self.data_dir = Path(data_dir)
        self.config = config or SchedulingConfig()
        self.max_iterations = max_iterations
        self.use_full_llm = use_full_llm
        
        # Initialize data loader
        self.data_loader = DataLoader(data_dir)
        
        # Initialize agents
        self.constraint_agent = ConstraintAgent(api_key)
        
        # Choose planner based on mode - now using tool-based for full LLM
        if use_full_llm:
            self.planner_agent = ToolBasedPlannerAgent(api_key)
        else:
            self.planner_agent = PlannerAgent(api_key, data_dir=str(self.data_dir))
            
        self.verification_agent = VerificationAgent(api_key)
        self.selection_agent = SelectionAgent(api_key)
        
        # Initialize agent memory for trajectory tracking
        self.memory = AgentMemory(self.data_dir / "memory")
        
        # State
        self.courses: dict[str, Course] = {}
        self.teachers: dict[str, Teacher] = {}
        self.constraints = []
        self.proposals: list[TimetableProposal] = []
        self.results: list[VerificationResult] = []
        
        # Output directory
        self.output_dir = self.data_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
    
    def run(self, verbose: bool = True) -> TimetableProposal:
        """
        Run the complete multi-agent scheduling workflow with memory and learning.
        """
        if verbose:
            print("=" * 60)
            print("🎓 Multi-Agent College Timetable Scheduling System")
            print("   with Memory & Iterative Learning (PlanGEN)")
            print("=" * 60)
        
        # Step 1: Load data
        if verbose:
            print("\n📚 Loading data...")
        self.data_loader.load_all()
        self.courses = self.data_loader.courses
        self.teachers = self.data_loader.teachers
        
        stats = self.data_loader.get_stats()
        if verbose:
            print(f"   Courses: {stats['total_courses']}")
            print(f"   Teachers: {stats['total_teachers']}")
            print(f"   Students: {stats['total_students']}")
            print(f"   Sessions to schedule: {stats['total_sessions_to_schedule']}")
            print(f"   Available slots: {self.config.total_slots * self.config.num_rooms}")
        
        # Step 2: Extract constraints
        if verbose:
            print("\n🔒 Extracting constraints...")
        self.constraints = self.constraint_agent.extract_constraints(
            self.courses, self.teachers, self.config
        )
        complexity = self.constraint_agent.analyze_constraint_density(
            self.courses, self.teachers, self.config
        )
        
        if verbose:
            print(f"   Hard constraints: {len([c for c in self.constraints if c.constraint_type == 'hard'])}")
            print(f"   Soft constraints: {len([c for c in self.constraints if c.constraint_type == 'soft'])}")
            print(f"   Complexity: {complexity['complexity_score']}")
        
        # Step 3: Check memory for previous learnings
        memory_context = self.memory.get_iteration_context()
        if "No previous iterations" not in memory_context:
            if verbose:
                print("\n🧠 Loading learnings from previous sessions...")
                print(f"   Found {len(self.memory.iterations)} previous iterations")
        
        # Step 4: Iterative generation with learning
        best_proposal = None
        best_score = -1.0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get memory context for this iteration
            iteration_context = self.memory.get_iteration_context()
            
            # Select algorithm (uses memory insights)
            algorithm = "llm_driven"  # Always use LLM-driven as per user request
            
            if verbose:
                print(f"   Algorithm: {algorithm}")
            
            # Generate proposal with memory context
            proposal = self.planner_agent.generate_proposal(
                self.courses, 
                self.teachers, 
                self.config,
                self.constraints, 
                algorithm=algorithm,
                previous_feedback=iteration_context if iteration > 0 else None
            )
            self.proposals.append(proposal)
            
            if verbose:
                print(f"   Generated {len(proposal.entries)} schedule entries")
            
            # Verify proposal
            result = self.verification_agent.verify(
                proposal, self.courses, self.teachers, self.config, self.constraints
            )
            self.results.append(result)
            
            # Record in memory for learning
            self.memory.record_iteration(
                iteration_id=iteration + 1,
                algorithm=algorithm,
                sessions_scheduled=len(proposal.entries),
                sessions_required=stats['total_sessions_to_schedule'],
                conflicts=result.conflicts,
                score=result.score,
                is_valid=result.is_valid,
                feedback=result.feedback,
                suggestions=result.suggestions
            )
            
            # Derive and record learnings
            learnings = self.memory.derive_learnings()
            for learning in learnings:
                self.memory.add_learning(learning)
            
            if verbose:
                coverage = len(proposal.entries) / stats['total_sessions_to_schedule'] * 100
                print(f"   Coverage: {coverage:.1f}%")
                print(f"   Score: {result.score:.3f}")
                print(f"   Valid: {result.is_valid}")
                if result.conflicts:
                    conflict_types = {}
                    for c in result.conflicts:
                        ct = c.get("type", "unknown")
                        conflict_types[ct] = conflict_types.get(ct, 0) + 1
                    print(f"   Conflicts: {conflict_types}")
            
            # Track best
            if result.score > best_score:
                best_score = result.score
                best_proposal = proposal
            
            # Early termination if valid
            if result.is_valid:
                if verbose:
                    print("\n✅ Found valid schedule!")
                self.memory.add_learning("Valid schedule achieved - record successful patterns")
                break
            
            # Add learning about this iteration's failures
            if result.conflicts:
                sample_conflicts = result.conflicts[:3]
                for c in sample_conflicts:
                    self.memory.add_learning(f"Avoid: {c.get('description', 'unknown conflict')[:80]}")
        
        if not best_proposal:
            if verbose:
                print("\n⚠️ No valid schedule found in max iterations")
            best_proposal = self.proposals[-1] if self.proposals else TimetableProposal(
                proposal_id="empty", entries=[]
            )
        
        # Step 5: Save outputs
        self._save_outputs(best_proposal, self.results[-1] if self.results else None)
        
        if verbose:
            print(f"\n📁 Output saved to: {self.output_dir}")
            print(f"   - timetable.csv")
            print(f"   - report.json")
            print(f"\n🧠 Memory saved with {len(self.memory.iterations)} iteration records")
            
            # Show learnings
            if self.memory.global_learnings:
                print("\n📝 Key Learnings:")
                for learning in self.memory.global_learnings[-5:]:
                    print(f"   - {learning[:80]}")
        
        return best_proposal
    
    def _save_outputs(self, proposal: TimetableProposal, result: Optional[VerificationResult]) -> None:
        """Save timetable and report to files."""
        # Save timetable as CSV
        csv_path = self.output_dir / "timetable.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Day", "Hour", "Course", "Batch", "Teacher", "Room", 
                "Session Type", "Students"
            ])
            
            sorted_entries = sorted(
                proposal.entries,
                key=lambda e: (
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(e.time_slot.day.value)
                    if e.time_slot.day.value in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] else 5,
                    e.time_slot.hour
                )
            )
            
            for entry in sorted_entries:
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
        
        # Save report as JSON
        report = {
            "generated_at": datetime.now().isoformat(),
            "proposal_id": proposal.proposal_id,
            "algorithm_used": proposal.algorithm_used,
            "total_entries": len(proposal.entries),
            "generation_time_ms": proposal.generation_time_ms,
            "config": {
                "days": [d.value for d in self.config.days],
                "hours": f"{self.config.start_hour}:00 - {self.config.end_hour}:00",
                "rooms": self.config.num_rooms,
                "room_capacity": self.config.room_capacity
            },
            "data_stats": self.data_loader.get_stats(),
            "scheduling_stats": self.planner_agent.get_scheduling_stats(proposal, self.courses),
            "memory": {
                "iterations_recorded": len(self.memory.iterations),
                "learnings": self.memory.global_learnings[-10:]
            }
        }
        
        if result:
            report["verification"] = {
                "is_valid": result.is_valid,
                "score": result.score,
                "conflicts_count": len(result.conflicts),
                "feedback": result.feedback,
                "suggestions": result.suggestions
            }
        
        report_path = self.output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    
    def clear_memory(self) -> None:
        """Clear agent memory to start fresh."""
        self.memory.clear()
        print("🧹 Agent memory cleared")
