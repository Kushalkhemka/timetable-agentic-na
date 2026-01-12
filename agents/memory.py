"""
Agent Memory - Tracks trajectories, failures, and learnings across iterations.
Enables iterative refinement following PlanGEN framework.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class IterationRecord:
    """Record of a single scheduling iteration."""
    iteration_id: int
    algorithm_used: str
    timestamp: str
    sessions_scheduled: int
    sessions_required: int
    coverage_percentage: float
    conflicts: list[dict]
    conflict_types: dict[str, int]  # e.g., {"teacher_conflict": 5, "room_conflict": 3}
    score: float
    is_valid: bool
    feedback: str
    suggestions: list[str]
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)


@dataclass  
class AgentAction:
    """Record of an agent's action."""
    agent_name: str
    action_type: str
    input_summary: str
    output_summary: str
    success: bool
    error_message: Optional[str] = None
    timestamp: str = ""


class AgentMemory:
    """
    Persistent memory for multi-agent scheduling system.
    Tracks:
    - Iteration trajectories and outcomes
    - Failure patterns and root causes
    - Successful strategies
    - Conflict patterns to avoid
    - Learning insights for refinement
    """
    
    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = memory_dir or Path("./memory")
        self.memory_dir.mkdir(exist_ok=True)
        
        self.iterations: list[IterationRecord] = []
        self.agent_actions: list[AgentAction] = []
        self.conflict_patterns: dict[str, list[dict]] = {}
        self.successful_patterns: list[dict] = []
        self.global_learnings: list[str] = []
        
        # Load existing memory if available
        self._load_memory()
    
    def record_iteration(
        self,
        iteration_id: int,
        algorithm: str,
        sessions_scheduled: int,
        sessions_required: int,
        conflicts: list[dict],
        score: float,
        is_valid: bool,
        feedback: str,
        suggestions: list[str]
    ) -> None:
        """Record the outcome of a scheduling iteration."""
        
        # Analyze conflict types
        conflict_types = {}
        for c in conflicts:
            ctype = c.get("type", "unknown")
            conflict_types[ctype] = conflict_types.get(ctype, 0) + 1
        
        # Identify what worked and what failed
        what_worked = []
        what_failed = []
        
        coverage = sessions_scheduled / sessions_required * 100 if sessions_required > 0 else 0
        
        if coverage > 90:
            what_worked.append(f"Achieved {coverage:.1f}% coverage")
        elif coverage < 50:
            what_failed.append(f"Poor coverage: only {coverage:.1f}%")
        
        if "teacher_conflict" in conflict_types:
            what_failed.append(f"{conflict_types['teacher_conflict']} teacher double-bookings")
        
        if "room_conflict" in conflict_types:
            what_failed.append(f"{conflict_types['room_conflict']} room conflicts")
        
        if "incomplete_coverage" in conflict_types:
            what_failed.append(f"{conflict_types['incomplete_coverage']} courses not fully scheduled")
        
        if is_valid:
            what_worked.append("Produced valid schedule")
        
        record = IterationRecord(
            iteration_id=iteration_id,
            algorithm_used=algorithm,
            timestamp=datetime.now().isoformat(),
            sessions_scheduled=sessions_scheduled,
            sessions_required=sessions_required,
            coverage_percentage=round(coverage, 2),
            conflicts=conflicts[:20],  # Keep only first 20 for memory efficiency
            conflict_types=conflict_types,
            score=score,
            is_valid=is_valid,
            feedback=feedback,
            suggestions=suggestions,
            what_worked=what_worked,
            what_failed=what_failed
        )
        
        self.iterations.append(record)
        
        # Track conflict patterns
        for c in conflicts[:50]:
            ctype = c.get("type", "unknown")
            if ctype not in self.conflict_patterns:
                self.conflict_patterns[ctype] = []
            self.conflict_patterns[ctype].append(c)
        
        self._save_memory()
    
    def record_agent_action(
        self,
        agent_name: str,
        action_type: str,
        input_summary: str,
        output_summary: str,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Record an action taken by an agent."""
        action = AgentAction(
            agent_name=agent_name,
            action_type=action_type,
            input_summary=input_summary[:200],  # Truncate for efficiency
            output_summary=output_summary[:200],
            success=success,
            error_message=error,
            timestamp=datetime.now().isoformat()
        )
        self.agent_actions.append(action)
    
    def add_learning(self, learning: str) -> None:
        """Add a global learning insight."""
        if learning not in self.global_learnings:
            self.global_learnings.append(learning)
            self._save_memory()
    
    def get_iteration_context(self) -> str:
        """Get context from previous iterations for LLM refinement."""
        if not self.iterations:
            return "No previous iterations."
        
        context_parts = ["=== PREVIOUS ITERATION HISTORY ===\n"]
        
        for record in self.iterations[-3:]:  # Last 3 iterations
            context_parts.append(f"""
Iteration {record.iteration_id} ({record.algorithm_used}):
- Coverage: {record.coverage_percentage}% ({record.sessions_scheduled}/{record.sessions_required})
- Score: {record.score}, Valid: {record.is_valid}
- Conflicts: {record.conflict_types}
- What failed: {', '.join(record.what_failed) if record.what_failed else 'None'}
- What worked: {', '.join(record.what_worked) if record.what_worked else 'None'}
- Suggestions: {', '.join(record.suggestions[:3]) if record.suggestions else 'None'}
""")
        
        # Add learnings
        if self.global_learnings:
            context_parts.append("\n=== KEY LEARNINGS ===")
            for learning in self.global_learnings[-5:]:
                context_parts.append(f"- {learning}")
        
        # Add common conflict patterns to avoid
        if self.conflict_patterns:
            context_parts.append("\n=== CONFLICT PATTERNS TO AVOID ===")
            for ctype, conflicts in list(self.conflict_patterns.items())[:3]:
                context_parts.append(f"- {ctype}: {len(conflicts)} occurrences")
                if conflicts:
                    sample = conflicts[0]
                    context_parts.append(f"  Example: {sample.get('description', 'N/A')[:100]}")
        
        return "\n".join(context_parts)
    
    def get_best_iteration(self) -> Optional[IterationRecord]:
        """Get the best performing iteration."""
        if not self.iterations:
            return None
        return max(self.iterations, key=lambda r: (r.is_valid, r.score, r.coverage_percentage))
    
    def derive_learnings(self) -> list[str]:
        """Analyze iterations and derive learnings."""
        learnings = []
        
        if len(self.iterations) < 2:
            return learnings
        
        # Compare algorithms
        algo_performance = {}
        for record in self.iterations:
            algo = record.algorithm_used
            if algo not in algo_performance:
                algo_performance[algo] = {"scores": [], "coverages": []}
            algo_performance[algo]["scores"].append(record.score)
            algo_performance[algo]["coverages"].append(record.coverage_percentage)
        
        best_algo = None
        best_avg_score = -1
        for algo, perf in algo_performance.items():
            avg = sum(perf["scores"]) / len(perf["scores"]) if perf["scores"] else 0
            if avg > best_avg_score:
                best_avg_score = avg
                best_algo = algo
        
        if best_algo:
            learnings.append(f"Algorithm '{best_algo}' performed best with avg score {best_avg_score:.3f}")
        
        # Identify recurring conflicts
        for ctype, conflicts in self.conflict_patterns.items():
            if len(conflicts) > 5:
                learnings.append(f"Recurring issue: {ctype} appeared {len(conflicts)} times")
        
        # Check improvement trend
        if len(self.iterations) >= 2:
            recent = self.iterations[-1]
            previous = self.iterations[-2]
            if recent.coverage_percentage > previous.coverage_percentage:
                learnings.append(f"Coverage improved: {previous.coverage_percentage}% -> {recent.coverage_percentage}%")
            if recent.score > previous.score:
                learnings.append(f"Score improved: {previous.score:.3f} -> {recent.score:.3f}")
        
        return learnings
    
    def _save_memory(self) -> None:
        """Persist memory to disk."""
        memory_file = self.memory_dir / "agent_memory.json"
        
        data = {
            "iterations": [asdict(r) for r in self.iterations[-10:]],  # Keep last 10
            "conflict_patterns": {k: v[-10:] for k, v in self.conflict_patterns.items()},
            "global_learnings": self.global_learnings[-20:],
            "agent_actions": [asdict(a) for a in self.agent_actions[-50:]]
        }
        
        with open(memory_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_memory(self) -> None:
        """Load memory from disk if exists."""
        memory_file = self.memory_dir / "agent_memory.json"
        
        if memory_file.exists():
            try:
                with open(memory_file) as f:
                    data = json.load(f)
                
                self.global_learnings = data.get("global_learnings", [])
                self.conflict_patterns = data.get("conflict_patterns", {})
                
                # Reconstruct iteration records
                for rec_dict in data.get("iterations", []):
                    self.iterations.append(IterationRecord(**rec_dict))
                    
            except Exception as e:
                print(f"[AgentMemory] Could not load memory: {e}")
    
    def clear(self) -> None:
        """Clear all memory."""
        self.iterations = []
        self.agent_actions = []
        self.conflict_patterns = {}
        self.successful_patterns = []
        self.global_learnings = []
        
        memory_file = self.memory_dir / "agent_memory.json"
        if memory_file.exists():
            memory_file.unlink()
