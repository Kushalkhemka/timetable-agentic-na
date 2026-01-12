# UCB Function and Prompting Strategies

## Upper Confidence Bound (UCB) Algorithm Selection

The `SelectionAgent` uses **UCB1** (Upper Confidence Bound) to adaptively choose scheduling algorithms. This balances **exploitation** (using what works) vs **exploration** (trying new approaches).

### UCB1 Formula

```
UCB_score = average_reward + sqrt(2 * ln(total_attempts) / algorithm_attempts)
```

| Term | Meaning |
|------|---------|
| `average_reward` | Mean score from previous runs of this algorithm |
| `exploration_bonus` | Bonus for under-explored algorithms |
| `total_attempts` | Total runs across all algorithms |

### How It Works

```python
# From selection_agent.py lines 58-92

def _ucb_select(self, iteration: int) -> str:
    for alg in self.ALGORITHMS:
        if stats["attempts"] == 0:
            return alg  # Unexplored gets priority
        
        avg_reward = stats["total_score"] / stats["attempts"]
        exploration = sqrt(2 * log(total_attempts) / stats["attempts"])
        ucb_score = avg_reward + exploration
```

### Example Selection Flow

| Iteration | Algorithm | Score | UCB Decision |
|-----------|-----------|-------|--------------|
| 1 | best_of_n | 0.0 | Initial choice for high complexity |
| 2 | greedy | 0.0 | UCB explores untried algorithm |
| 3 | random | 0.0 | UCB explores untried algorithm |
| 4 | llm_guided | 0.3 | UCB explores last untried |
| 5 | llm_guided | - | UCB selects highest scoring |

---

## Prompting Strategies

### 1. Strategy Prompt (PlannerAgent)

**Purpose:** Get high-level scheduling strategy from LLM

```
You are a scheduling expert. Analyze this scheduling problem and provide a strategy.

PROBLEM:
- 158 courses to schedule
- 79 theory-only, 79 with labs
- 17 large courses (3+ batches)
- 1064 total sessions needed
- 1080 available slots

Respond with ONLY a JSON object:
{"approach": "priority_large_first" or "distribute_evenly" or "cluster_by_teacher",
 "priority_courses": [...],
 "prefer_morning_theory": true/false,
 "max_consecutive_hours": 3 or 4,
 "reasoning": "..."}
```

**Why this prompt works:**
- Gives concrete problem metrics
- Limits output options (3 approaches)
- Requests structured JSON
- Asks for reasoning (improves quality)

---

### 2. Conflict Resolution Prompt

**Purpose:** Get suggestions when scheduling fails

```
A scheduling system produced conflicts:
{conflict_details}

Current schedule has {N} sessions.
Suggest 3 specific strategies to resolve these conflicts.
Focus on practical reordering or reassignment suggestions.
Keep response under 200 words.
```

**Why this prompt works:**
- Provides context (what failed)
- Limits to 3 specific suggestions
- Word limit prevents rambling
- "Practical" focuses on actionable advice

---

### Prompt Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Structured output** | Request JSON with specific keys |
| **Constraints** | Limit options ("A or B or C") |
| **Context** | Include problem metrics |
| **Word limits** | "Under 200 words" |
| **Temperature** | Low (0.3) for consistency |
