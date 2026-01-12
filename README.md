# Multi-Agent Timetable Scheduling System

A PlanGEN-inspired multi-agent system for university timetable scheduling using LLM-guided algorithms.

## Overview

This system schedules **1064 sessions** (816 theory + 248 lab) across **30 rooms** with **zero teacher/room conflicts** and **minimized student conflicts**.

### Results

| Metric | Value |
|--------|-------|
| **Coverage** | 100% (1064/1064 sessions) |
| **Teacher Conflicts** | 0 |
| **Room Conflicts** | 0 |
| **Student Conflicts** | 6,182 (reduced from 13,583) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                              │
│  Coordinates multi-agent workflow                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────────┐ ┌────────────┐ ┌──────────────┐
│ SELECTION     │ │ PLANNER    │ │ VERIFICATION │
│ AGENT (LLM)   │ │ AGENT      │ │ AGENT        │
│               │ │            │ │              │
│ UCB algorithm │ │ Hybrid:    │ │ Validates    │
│ picks best    │ │ LLM+Greedy │ │ constraints  │
│ strategy      │ │            │ │              │
└───────────────┘ └────────────┘ └──────────────┘
```

---

## The Hybrid Approach

### Why Hybrid?

| Approach | Pros | Cons |
|----------|------|------|
| **Pure LLM** | Intelligent decisions | Too slow (100+ calls) |
| **Pure Algorithm** | Fast | No learning/adaptation |
| **Hybrid** | Best of both | Complexity |

### How It Works

#### Phase 1: LLM Strategy Selection
```python
# The LLM analyzes the problem and picks a strategy
strategy = llm.call("""
  Given 158 courses, 59 teachers, 30 rooms:
  Choose scheduling approach:
  - labs_first: Schedule labs before theory
  - theory_first: Schedule theory before labs
  - most_constrained_first: Hardest courses first
""")
# Result: "labs_first" with conflict-aware ordering
```

#### Phase 2: Greedy Scheduling with Conflict Minimization
```python
# 1. Sort by conflict density (most constrained first)
courses.sort(key=lambda c: -conflict_count[c])

# 2. For each course-batch:
for course in courses:
    best_slot = None
    min_conflicts = infinity
    
    # 3. Find slot with MINIMUM student conflicts
    for slot in all_slots:
        if teacher_available(slot) and room_available(slot):
            conflicts = count_student_conflicts(slot, course)
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_slot = slot
    
    # 4. Assign to best slot
    assign(course, best_slot)
```

#### Phase 3: Teacher Reassignment (if needed)
```python
# If a course couldn't be scheduled due to teacher conflict:
for incomplete_course in unscheduled:
    alternative_teacher = find_available_teacher(course)
    if alternative_teacher:
        reassign(incomplete_course, alternative_teacher)
```

---

## Key Components

### 1. Selection Agent (`agents/selection_agent.py`)
Uses **UCB (Upper Confidence Bound)** algorithm to pick the best scheduling strategy based on historical performance.

```python
# UCB Formula
score = mean_score + sqrt(2 * log(total_trials) / trials_for_this_strategy)
```

### 2. Planner Agent (`agents/planner_agent.py`)
The hybrid scheduler with:
- LLM-guided strategy selection
- Greedy slot assignment
- Conflict-aware optimization
- Teacher reassignment fallback

### 3. Conflict-Aware Planner (`agents/conflict_aware_planner.py`)
Improved version that:
- Weights conflicts by student count
- Uses graph coloring heuristics (DSatur)
- Achieves 54.5% conflict reduction

### 4. Verification Agent (`agents/verification_agent.py`)
Validates the schedule against:
- Room conflicts
- Teacher conflicts
- Student conflicts
- Coverage requirements

### 5. Student Conflict Matrix (`utils/student_conflicts.py`)
Pre-computes which course-batches share students:
```python
# 9,546 conflict pairs identified
# Each pair weighted by number of shared students
# Max: EC401-B1 vs EC403-B1 = 82 shared students
```

---

## Hard Constraints (Never Violated)

| Constraint | Implementation |
|------------|----------------|
| One teacher per slot | `teacher_schedule[(day, hour)].add(teacher)` |
| One room per slot | `room_schedule[(day, hour, room)] = True` |
| Labs in LAB rooms | `rooms = lab_rooms if type == "lab" else regular_rooms` |
| All hours scheduled | Loop until `hours_scheduled == hours_needed` |

---

## Soft Constraints (Minimized)

| Constraint | Approach |
|------------|----------|
| Student conflicts | Pick slots with minimum existing conflicts |
| Course spreading | (Not implemented) |
| Lunch breaks | (Not implemented) |

---

## Algorithm: DSatur (Degree of Saturation)

The scheduling problem is equivalent to **graph coloring**:
- **Nodes**: Course-batches (235)
- **Edges**: Student conflicts (9,546 pairs)
- **Colors**: Time slots (40 = 5 days × 8 hours)

DSatur orders nodes by "saturation degree" - how many different colors are already used by neighbors. This prevents getting stuck.

```
1. Pick uncolored node with highest saturation
2. Assign lowest-numbered available color
3. Repeat until all colored
```

For timetabling:
```
1. Pick unscheduled course with most conflicts
2. Assign slot with minimum additional conflicts
3. Repeat until all scheduled
```

---

## Why Greedy Beats Genetic Here

| Factor | Greedy | Genetic |
|--------|--------|---------|
| **Slack** | 12% extra slots → easy | Wastes time exploring |
| **Constraints** | Always feasible | Often violates |
| **Speed** | 87ms | Hours |
| **Determinism** | Same result each time | Random |

---

## Usage

### Basic Run
```bash
python main.py --rooms 30 --max-iter 1
```

### Conflict-Aware Run
```bash
python -c "from agents.conflict_aware_planner import run_conflict_aware_scheduler; run_conflict_aware_scheduler('.')"
```

### Output
```
output/
├── timetable.csv          # Main schedule
├── teacher_timetables/    # Per-teacher schedules
└── room_timetables/       # Per-room schedules
```

---

## Files

```
TT/
├── agents/
│   ├── base_agent.py           # LLM wrapper (Gemini 2.0 Flash)
│   ├── planner_agent.py        # Hybrid scheduler
│   ├── conflict_aware_planner.py # Improved scheduler
│   ├── selection_agent.py      # UCB strategy selector
│   └── verification_agent.py   # Constraint validator
├── utils/
│   ├── data_loader.py          # CSV parsing
│   └── student_conflicts.py    # Conflict matrix
├── scheduler/
│   └── orchestrator.py         # Multi-agent coordination
├── models/
│   └── data_models.py          # Pydantic models
├── main.py                     # CLI entry point
└── requirements.txt
```

---

## Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_api_key_here
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rooms` | 30 | Number of rooms (21 regular + 7 labs) |
| `--max-iter` | 5 | Max refinement iterations |
| `--algorithm` | hybrid | Scheduling algorithm |

---

## Performance

| Metric | Value |
|--------|-------|
| **Scheduling Time** | ~90ms |
| **LLM Calls** | 1-2 |
| **Total Sessions** | 1,064 |
| **Conflict Reduction** | 54.5% |
