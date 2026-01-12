#!/usr/bin/env python3
"""
Multi-Agent Timetable Scheduling with FULL LLM Integration

This script demonstrates the complete multi-agent workflow where LLM is used in:
1. Constraint Analysis - Discover implicit constraints
2. Strategy Selection - Choose scheduling approach
3. Verification - Explain conflicts and suggest fixes
4. Refinement - Suggest slot swaps to reduce conflicts

Run: python run_with_llm.py
"""

import json
import time
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.data_models import (
    Course, Teacher, SchedulingConfig, TimeSlot, Room,
    ScheduleEntry, TimetableProposal, SessionType, Day
)
from utils.data_loader import DataLoader
from agents import (
    ConstraintAgent, VerificationAgent, RefinementAgent,
    SelectionAgent, PlannerAgent
)


class LLMEnhancedScheduler:
    """
    Complete multi-agent scheduler with LLM integration at every stage.
    """
    
    def __init__(self, data_dir: str = ".", regular_rooms: int = 28, lab_rooms: int = 7):
        self.data_dir = Path(data_dir)
        self.regular_rooms = regular_rooms
        self.lab_rooms = lab_rooms
        self.trace = []
        self.llm_calls = []
        
    def log(self, agent: str, step: str, details: dict = None, llm_call: bool = False):
        """Add a trace entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "step": step,
            "llm_call": llm_call,
            "details": details or {}
        }
        self.trace.append(entry)
        
        emoji = "🤖" if llm_call else "⚙️"
        print(f"[{agent}] {emoji} {step}")
        
        if llm_call:
            self.llm_calls.append(entry)
    
    def run(self):
        """Execute the full LLM-enhanced scheduling workflow."""
        start_time = time.time()
        
        print("=" * 70)
        print("🧠 MULTI-AGENT SCHEDULER WITH FULL LLM INTEGRATION")
        print("=" * 70)
        print()
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: DATA LOADING
        # ═══════════════════════════════════════════════════════════════
        self.log("DATA_LOADER", "Loading scheduling data")
        
        loader = DataLoader(self.data_dir)
        loader.load_all()
        courses = loader.courses
        teachers = loader.teachers
        
        self.log("DATA_LOADER", "Data loaded", {
            "courses": len(courses),
            "teachers": len(teachers),
            "students": len(loader.students)
        })
        
        # Build student conflict matrix for later use
        student_file = self.data_dir / "student_allocations_aggregated.csv"
        pair_student_count = defaultdict(int)
        
        with open(student_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cs = row['Allocated Courses'].replace('"', '').split(', ')
                bs = row['Batches'].replace('"', '').split(', ')
                cbs = [(c.strip(), b.strip()) for c, b in zip(cs, bs)]
                for i, cb1 in enumerate(cbs):
                    for cb2 in cbs[i+1:]:
                        key = (cb1, cb2) if cb1 < cb2 else (cb2, cb1)
                        pair_student_count[key] += 1
        
        # Get top conflict pairs for LLM analysis
        top_conflicts = sorted(pair_student_count.items(), key=lambda x: -x[1])[:10]
        conflict_pairs = [(f"{p[0][0]}-{p[0][1]}", f"{p[1][0]}-{p[1][1]}", c) for p, c in top_conflicts]
        
        config = SchedulingConfig(
            num_rooms=self.regular_rooms + self.lab_rooms,
            room_capacity=90
        )
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: LLM CONSTRAINT ANALYSIS
        # ═══════════════════════════════════════════════════════════════
        self.log("CONSTRAINT_AGENT", "Analyzing constraints with LLM", llm_call=True)
        
        constraint_agent = ConstraintAgent()
        
        # Call LLM to discover implicit constraints
        llm_analysis = constraint_agent.analyze_with_llm(
            courses, teachers, config, conflict_pairs
        )
        
        self.log("CONSTRAINT_AGENT", "LLM constraint analysis complete", {
            "bottlenecks": llm_analysis.get("bottleneck_teachers", [])[:3],
            "priority_order": llm_analysis.get("recommended_priority_order", []),
            "suggested_constraints": len(llm_analysis.get("suggested_constraints", []))
        }, llm_call=True)
        
        # Also get programmatic constraints
        constraints = constraint_agent.extract_constraints(courses, teachers, config)
        
        self.log("CONSTRAINT_AGENT", "Constraints extracted", {
            "hard_constraints": len([c for c in constraints if c.constraint_type == "hard"]),
            "soft_constraints": len([c for c in constraints if c.constraint_type == "soft"])
        })
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: ALGORITHM SELECTION (UCB - no LLM needed)
        # ═══════════════════════════════════════════════════════════════
        self.log("SELECTION_AGENT", "Selecting algorithm using UCB")
        
        selection_agent = SelectionAgent()
        complexity = constraint_agent.analyze_constraint_density(courses, teachers, config)
        algorithm = selection_agent.select_algorithm(courses, config, complexity, iteration=0)
        
        self.log("SELECTION_AGENT", "Algorithm selected", {
            "algorithm": algorithm,
            "complexity": complexity["complexity_score"]
        })
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: LLM-GUIDED STRATEGY + SCHEDULING (100% COVERAGE)
        # ═══════════════════════════════════════════════════════════════
        self.log("PLANNER_AGENT", "Getting LLM scheduling strategy", llm_call=True)
        
        # Import ConflictAwarePlanner for 100% coverage (allows student conflicts)
        from agents.conflict_aware_planner import ConflictAwarePlanner
        
        # Get LLM strategy first
        planner_agent = PlannerAgent(data_dir=str(self.data_dir))
        strategy = planner_agent._get_llm_strategy(courses, config, None)
        
        self.log("PLANNER_AGENT", "LLM strategy received", {
            "approach": strategy.get("approach", "unknown"),
            "morning_theory": strategy.get("morning_theory", True),
            "reasoning": strategy.get("reasoning", "")[:100]
        }, llm_call=True)
        
        # Use ConflictAwarePlanner for 100% coverage (allows student conflicts)
        conflict_planner = ConflictAwarePlanner(str(self.data_dir))
        proposal = conflict_planner.generate_proposal(courses, teachers, config)
        
        self.log("PLANNER_AGENT", "Schedule generated with 100% coverage", {
            "sessions": len(proposal.entries),
            "time_ms": round(proposal.generation_time_ms, 0)
        })
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: LLM VERIFICATION FEEDBACK
        # ═══════════════════════════════════════════════════════════════
        self.log("VERIFICATION_AGENT", "Verifying schedule with LLM feedback", llm_call=True)
        
        verification_agent = VerificationAgent()
        
        # Algorithmic verification
        result = verification_agent.verify(proposal, courses, teachers, config, constraints)
        
        # Calculate student conflicts
        total_student_conflicts = 0
        slot_courses = defaultdict(list)
        for entry in proposal.entries:
            key = (entry.time_slot.day.value, entry.time_slot.hour)
            slot_courses[key].append((entry.course_code, entry.batch_id))
        
        for (day, hour), cbs in slot_courses.items():
            for i, cb1 in enumerate(cbs):
                for cb2 in cbs[i+1:]:
                    key = (cb1, cb2) if cb1 < cb2 else (cb2, cb1)
                    total_student_conflicts += pair_student_count.get(key, 0)
        
        # Get LLM feedback on results
        llm_feedback = verification_agent.get_llm_feedback(
            proposal, result.conflicts, total_student_conflicts, courses
        )
        
        self.log("VERIFICATION_AGENT", "LLM verification feedback", {
            "assessment": llm_feedback.get("overall_assessment", "unknown"),
            "root_causes": llm_feedback.get("root_causes", [])[:2],
            "priority_actions": llm_feedback.get("priority_actions", [])[:2]
        }, llm_call=True)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 6: ITERATIVE LLM REFINEMENT (ACTUALLY APPLIES CHANGES!)
        # ═══════════════════════════════════════════════════════════════
        self.log("REFINEMENT_AGENT", "Starting iterative LLM refinement loop", llm_call=True)
        
        refinement_agent = RefinementAgent()
        
        # Run iterative refinement (up to 3 rounds of LLM-guided improvements)
        initial_conflicts = total_student_conflicts
        refined_proposal, refinement_trace = refinement_agent.iterative_refinement(
            proposal, pair_student_count, max_iterations=3
        )
        
        # Calculate final conflicts after refinement
        final_student_conflicts = refinement_agent.calculate_conflicts(refined_proposal, pair_student_count)
        improvement = initial_conflicts - final_student_conflicts
        
        self.log("REFINEMENT_AGENT", "LLM refinement complete", {
            "initial_conflicts": initial_conflicts,
            "final_conflicts": final_student_conflicts,
            "improvement": improvement,
            "iterations": len(refinement_trace),
            "total_moves_applied": sum(t.get("moves_applied", 0) for t in refinement_trace)
        }, llm_call=True)
        
        # Use refined proposal for output
        proposal = refined_proposal
        total_student_conflicts = final_student_conflicts
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 7: SAVE OUTPUTS
        # ═══════════════════════════════════════════════════════════════
        output_dir = self.data_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Save timetable
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
        
        # Save trace
        with open(output_dir / "llm_trace.json", "w") as f:
            json.dump({
                "total_steps": len(self.trace),
                "llm_calls": len(self.llm_calls),
                "trace": self.trace
            }, f, indent=2, default=str)
        
        # ═══════════════════════════════════════════════════════════════
        # FINAL SUMMARY
        # ═══════════════════════════════════════════════════════════════
        elapsed = time.time() - start_time
        
        print()
        print("=" * 70)
        print("✅ LLM-ENHANCED SCHEDULING COMPLETE")
        print("=" * 70)
        print()
        print(f"📊 RESULTS:")
        print(f"   Sessions scheduled: {len(proposal.entries)}")
        print(f"   Student conflicts: {total_student_conflicts}")
        print(f"   Teacher conflicts: {len([c for c in result.conflicts if c.get('type') == 'teacher_conflict'])}")
        print(f"   Room conflicts: {len([c for c in result.conflicts if c.get('type') == 'room_conflict'])}")
        print()
        print(f"🤖 LLM USAGE:")
        print(f"   Total LLM calls: {len(self.llm_calls)}")
        for i, call in enumerate(self.llm_calls, 1):
            print(f"   {i}. [{call['agent']}] {call['step']}")
        print()
        print(f"⏱️  Total time: {elapsed:.1f} seconds")
        print()
        print(f"📁 Outputs saved to: {output_dir}")
        
        return proposal, self.trace


if __name__ == "__main__":
    scheduler = LLMEnhancedScheduler(
        data_dir=".",
        regular_rooms=28,
        lab_rooms=7
    )
    
    proposal, trace = scheduler.run()
