"""
Selection Agent - Adaptively selects the best scheduling algorithm.
"""
from typing import Optional
from collections import defaultdict

from .base_agent import BaseAgent
from models.data_models import SchedulingConfig, Course, VerificationResult


class SelectionAgent(BaseAgent):
    """
    Selects optimal algorithm based on problem complexity and previous performance.
    Following PlanGEN: uses UCB-like strategy for algorithm selection.
    """
    
    ALGORITHMS = ["greedy", "random", "best_of_n", "llm_guided"]
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        
        # Track algorithm performance
        self._algorithm_stats = defaultdict(lambda: {
            "attempts": 0,
            "successes": 0,
            "total_score": 0.0,
            "avg_time_ms": 0.0
        })
    
    def select_algorithm(
        self,
        courses: dict[str, Course],
        config: SchedulingConfig,
        complexity_analysis: dict,
        iteration: int = 0
    ) -> str:
        """
        Select the best algorithm based on problem complexity and history.
        Uses Upper Confidence Bound (UCB) inspired selection.
        """
        complexity = complexity_analysis.get("complexity_score", "medium")
        
        # First iteration: select based on complexity
        if iteration == 0:
            if complexity == "low":
                selected = "greedy"
            elif complexity == "medium":
                selected = "best_of_n"
            else:
                selected = "best_of_n"  # Start with best_of_n for high complexity
        else:
            # Use performance history for subsequent iterations
            selected = self._ucb_select(iteration)
        
        self.log(f"Selected algorithm: {selected} (complexity={complexity}, iteration={iteration})")
        return selected
    
    def _ucb_select(self, iteration: int) -> str:
        """Select algorithm using UCB1 formula."""
        import math
        
        total_attempts = sum(
            self._algorithm_stats[alg]["attempts"] 
            for alg in self.ALGORITHMS
        )
        
        if total_attempts == 0:
            return "greedy"
        
        best_alg = "greedy"
        best_score = -float("inf")
        
        for alg in self.ALGORITHMS:
            stats = self._algorithm_stats[alg]
            
            if stats["attempts"] == 0:
                # Unexplored algorithm gets high priority
                return alg
            
            # Average reward
            avg_reward = stats["total_score"] / stats["attempts"]
            
            # Exploration bonus (UCB1)
            exploration = math.sqrt(2 * math.log(total_attempts) / stats["attempts"])
            
            ucb_score = avg_reward + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_alg = alg
        
        return best_alg
    
    def update_stats(self, algorithm: str, result: VerificationResult, time_ms: float) -> None:
        """Update algorithm performance statistics."""
        stats = self._algorithm_stats[algorithm]
        
        stats["attempts"] += 1
        stats["total_score"] += result.score
        
        if result.is_valid:
            stats["successes"] += 1
        
        # Running average for time
        n = stats["attempts"]
        stats["avg_time_ms"] = (stats["avg_time_ms"] * (n - 1) + time_ms) / n
        
        self.log(f"Updated stats for {algorithm}: attempts={stats['attempts']}, avg_score={stats['total_score']/stats['attempts']:.3f}")
    
    def get_recommendation(
        self,
        complexity_analysis: dict,
        previous_results: list[VerificationResult]
    ) -> dict:
        """
        Get algorithm recommendation with explanation.
        """
        complexity = complexity_analysis.get("complexity_score", "medium")
        slot_util = complexity_analysis.get("slot_utilization_ratio", 0.5)
        
        recommendation = {
            "primary_algorithm": "best_of_n",
            "fallback_algorithm": "llm_guided",
            "max_iterations": 5,
            "reasoning": ""
        }
        
        if complexity == "low" and slot_util < 0.4:
            recommendation["primary_algorithm"] = "greedy"
            recommendation["fallback_algorithm"] = "random"
            recommendation["max_iterations"] = 3
            recommendation["reasoning"] = "Low complexity problem with plenty of available slots. Greedy should work well."
        
        elif complexity == "medium":
            recommendation["primary_algorithm"] = "best_of_n"
            recommendation["fallback_algorithm"] = "greedy"
            recommendation["max_iterations"] = 5
            recommendation["reasoning"] = "Medium complexity. Best-of-N provides good solutions with moderate compute."
        
        else:  # high complexity
            recommendation["primary_algorithm"] = "best_of_n"
            recommendation["fallback_algorithm"] = "llm_guided"
            recommendation["max_iterations"] = 10
            recommendation["reasoning"] = "High complexity problem with tight constraints. May need multiple iterations and LLM guidance."
        
        # Adjust based on previous results
        if previous_results:
            avg_score = sum(r.score for r in previous_results) / len(previous_results)
            if avg_score < 0.5:
                recommendation["fallback_algorithm"] = "llm_guided"
                recommendation["reasoning"] += " Previous attempts had low scores, recommend LLM guidance."
        
        return recommendation
    
    def get_stats_summary(self) -> str:
        """Get a summary of algorithm performance."""
        lines = ["=== Algorithm Performance Summary ==="]
        
        for alg in self.ALGORITHMS:
            stats = self._algorithm_stats[alg]
            if stats["attempts"] > 0:
                avg_score = stats["total_score"] / stats["attempts"]
                success_rate = stats["successes"] / stats["attempts"] * 100
                lines.append(
                    f"{alg}: {stats['attempts']} attempts, "
                    f"avg_score={avg_score:.3f}, "
                    f"success_rate={success_rate:.1f}%, "
                    f"avg_time={stats['avg_time_ms']:.1f}ms"
                )
            else:
                lines.append(f"{alg}: no attempts yet")
        
        return "\n".join(lines)
