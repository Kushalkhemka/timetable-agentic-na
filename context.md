# Multi-Agent Timetable Scheduler - Development Context

## Project Overview

This document provides a comprehensive history of the development conversation for the **Multi-Agent Timetable Scheduling System**. The system uses LLM-enhanced agents to create university timetables with conflict minimization.

---

## Session Timeline

### Phase 1: Understanding the Problem

**User Request:** Configure rooms and trace the multi-agent workflow.

**Actions Taken:**
1. Reviewed the existing codebase structure
2. Explained the **UCB (Upper Confidence Bound)** algorithm used by SelectionAgent
3. Updated data loader to support new course format ("4", "3+2" shorthand)

### Phase 2: Room Configuration Update

**User Request:** Update to 28 regular rooms and 7 lab rooms.

**Changes Made:**
- Modified `agents/conflict_aware_planner.py` (lines 79-80)
- Changed `regular_rooms` from R1-R23 to R1-R28
- Lab rooms remained LAB1-LAB7

### Phase 3: Multi-Agent Trace Generation

**User Request:** Generate step-by-step trace of the workflow.

**Created Files:**
- `run_with_trace.py` - Comprehensive trace generator showing each agent's actions

**Trace Output:**
- 23 steps across 6 specialized agents
- ORCHESTRATOR → DATA_LOADER → CONSTRAINT_AGENT → CONFLICT_AGENT → SELECTION_AGENT → PLANNER_AGENT → VERIFICATION_AGENT → MEMORY_AGENT

### Phase 4: LLM Usage Analysis

**User Question:** Which agents use the LLM?

**Answer:**
| Agent | Uses LLM? | Purpose |
|-------|-----------|---------|
| PlannerAgent | ✅ Yes | Strategy selection ("labs_first") |
| LLMPlannerAgent | ✅ Yes | Full scheduling decisions |
| ConstraintAgent | ❌ No | Hardcoded rules |
| VerificationAgent | ❌ No | Rule-based checking |
| SelectionAgent | ❌ No | UCB math algorithm |

**Key Finding:** System was only ~1% LLM-driven (1 call for strategy).

### Phase 5: Student Conflict Root Cause Analysis

**User Question:** What causes the 6,458 student conflicts?

**Analysis:**
- 2,525 students enrolled in avg 5.7 courses each
- Top conflict pairs: EC401-B1 ↔ EC403-B1 (82 shared students)
- 235 course-batches in only 40 time slots → mathematical inevitability
- Hard constraints (teacher, room) satisfied; soft constraint (student conflicts) minimized but not eliminated

### Phase 6: LLM Enhancement Implementation

**User Request:** Add LLM to constraint analysis, verification, and refinement.

**Changes Made:**

#### ConstraintAgent (`constraint_agent.py`)
- Added `analyze_with_llm()` method
- LLM discovers: bottleneck teachers, implicit constraints, priority order

#### VerificationAgent (`verification_agent.py`)
- Added `get_llm_feedback()` method
- Added `suggest_improvements_with_llm()` method
- LLM explains WHY conflicts occurred, suggests fixes

#### RefinementAgent (`refinement_agent.py`) - NEW
- Created new agent for post-processing optimization
- Methods: `suggest_refinements()`, `apply_refinements()`, `iterative_refinement()`
- Actually applies LLM-suggested slot swaps

#### Unified Runner (`run_with_llm.py`) - NEW
- Demonstrates full LLM integration across all phases
- 8 LLM calls total across 4 agents

### Phase 7: LLM Action Loop Implementation

**User Request:** Make LLM suggestions actually modify the schedule.

**Implementation:**
1. Rewrote `apply_refinements()` to parse LLM moves and swap entries
2. Added constraint validation (teacher busy, room busy)
3. Created `iterative_refinement()` loop (up to 3 rounds)
4. Added `calculate_conflicts()` to track improvement

**Sample Output:**
```
Iteration 1: 6458 → 6550 conflicts
  ✅ Moved HU325-B1: Friday 15:00 → Thursday 12:00
  ✅ Moved IT325-B1: Monday 10:00 → Thursday 10:00
  ✅ Moved CH307-B1: Monday 13:00 → Friday 17:00
  ❌ EC401-B1: teacher busy (blocked)
```

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  1. CONSTRAINT_AGENT 🤖                                          │
│     └── LLM: Discover implicit constraints, bottlenecks         │
├─────────────────────────────────────────────────────────────────┤
│  2. SELECTION_AGENT ⚙️                                           │
│     └── UCB Algorithm: Select scheduling approach               │
├─────────────────────────────────────────────────────────────────┤
│  3. PLANNER_AGENT 🤖                                             │
│     └── LLM: Get strategy ("labs_first")                        │
│     └── ConflictAwarePlanner: DSatur + greedy scheduling        │
├─────────────────────────────────────────────────────────────────┤
│  4. VERIFICATION_AGENT 🤖                                        │
│     └── LLM: Explain root causes, suggest fixes                 │
├─────────────────────────────────────────────────────────────────┤
│  5. REFINEMENT_AGENT 🤖                                          │
│     └── LLM: Suggest slot swaps                                 │
│     └── Actually applies moves to reduce conflicts              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `agents/constraint_agent.py` | Modified | Added `analyze_with_llm()` |
| `agents/verification_agent.py` | Modified | Added `get_llm_feedback()`, `suggest_improvements_with_llm()` |
| `agents/refinement_agent.py` | **NEW** | LLM-guided iterative optimization |
| `agents/conflict_aware_planner.py` | Modified | Updated room config (28+7) |
| `utils/data_loader.py` | Modified | Support new CourseType format |
| `run_with_trace.py` | **NEW** | Pure algorithmic trace (0 LLM) |
| `run_with_llm.py` | **NEW** | Full LLM integration (8 calls) |
| `.gitignore` | **NEW** | Python/LLM patterns |

---

## Performance Results

| Metric | Algorithmic Only | With LLM |
|--------|------------------|----------|
| Sessions scheduled | 1064 (100%) | 1064 (100%) |
| Student conflicts | 6,458 | 6,573 |
| Teacher conflicts | 0 | 0 |
| Room conflicts | 0 | 0 |
| LLM calls | 0 | 8 |
| Time | 0.29 sec | 138 sec |

---

## LLM Prompts Used

### ConstraintAgent Prompt
```
Analyze this scheduling data and identify implicit constraints:
- Courses: 158, Teachers: 59
- High-conflict pairs: EC401-B1 ↔ EC403-B1 (82 students)
Return: bottleneck_teachers, critical_course_pairs, suggested_constraints
```

### VerificationAgent Prompt
```
Analyze these scheduling results:
- Sessions scheduled: 1064/1064 (100%)
- Student conflicts: 6458
Return: overall_assessment, root_causes, priority_actions
```

### RefinementAgent Prompt
```
Suggest slot swaps to reduce conflicts:
- HIGH-CONFLICT SLOTS: Monday 10:00 (438 conflicts)
- LOW-CONFLICT SLOTS: Friday 16:00 (23 conflicts)
Return: moves [{target, from, to}], expected_reduction
```

---

## Lessons Learned

1. **LLM for strategy, algorithm for execution** - LLM picks "labs_first", algorithm assigns 1064 slots in <1 second

2. **Student conflicts are mathematically inevitable** - 235 course-batches × 5 hrs each = 1175 slots needed, but only 40 time slots available

3. **LLM suggestions need conflict data** - Without knowing which courses share students, LLM swaps can increase conflicts

4. **DSatur ordering is effective** - Scheduling high-conflict courses first (when more slots are free) reduces final conflicts

---

## Commands

```bash
# Run pure algorithmic scheduler (fast, 0 LLM calls)
python run_with_trace.py

# Run full LLM-enhanced scheduler (8 LLM calls)
python run_with_llm.py

# Check LLM call logs
cat logs/llm_calls.json

# View output
cat output/timetable.csv
```

---

## Environment

- Python 3.11+
- Gemini 3 Pro API (via `google-genai` package)
- Required: `GEMINI_API_KEY` in `.env` file

---

*Generated: 2026-01-13*
