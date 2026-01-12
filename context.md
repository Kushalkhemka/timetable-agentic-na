# Multi-Agent Timetable Scheduler - Complete Development Context

> **Repository:** https://github.com/Kushalkhemka/timetable-agentic-na
> **Date:** 2026-01-13
> **Session Duration:** ~2 hours

---

## Table of Contents

1. [Session Overview](#session-overview)
2. [Phase 1: Initial Setup & Data Configuration](#phase-1-initial-setup--data-configuration)
3. [Phase 2: UCB Algorithm Explanation](#phase-2-ucb-algorithm-explanation)
4. [Phase 3: Multi-Agent Trace Generation](#phase-3-multi-agent-trace-generation)
5. [Phase 4: Student Conflict Analysis](#phase-4-student-conflict-analysis)
6. [Phase 5: LLM Usage Analysis](#phase-5-llm-usage-analysis)
7. [Phase 6: LLM Enhancement Implementation](#phase-6-llm-enhancement-implementation)
8. [Phase 7: LLM Action Loop Implementation](#phase-7-llm-action-loop-implementation)
9. [Final State & Files](#final-state--files)
10. [Technical Reference](#technical-reference)

---

## Session Overview

### User's Main Objectives

1. Configure scheduling with **28 regular rooms + 7 lab rooms**
2. Generate a **step-by-step trace** of the multi-agent workflow
3. Understand which agents use **LLM** and for what purpose
4. Make LLM insights **actually modify the schedule** (not just log)
5. Push code to GitHub with comprehensive documentation

### Key Outcomes

| Metric | Before | After |
|--------|--------|-------|
| LLM Calls | 1 | 8 |
| Agents with LLM | 1 (PlannerAgent) | 4 (Constraint, Planner, Verification, Refinement) |
| LLM insights applied | 0 (logged only) | 7 slot swaps applied |
| Documentation | Minimal | Full context.md |

---

## Phase 1: Initial Setup & Data Configuration

### User Request
> "Configure Rooms, Trace Workflow"

### Understanding the Existing System

The project already had a multi-agent architecture:
- **SchedulingOrchestrator** - Coordinates workflow
- **DataLoader** - Loads CSV files
- **ConstraintAgent** - Extracts constraints (hardcoded)
- **SelectionAgent** - UCB algorithm to pick scheduling approach
- **PlannerAgent** - LLM-guided strategy selection
- **VerificationAgent** - Rule-based conflict counting
- **AgentMemory** - JSON storage for learnings

### New Course Mapping File

User provided `course_batch_teachers (2).csv` with a **new format**:

**Old Format:**
```csv
CourseCode,CourseType,BatchID,TeacherName
CO401,Theory 4hr,B1,Dr. Sharma
CO403,Theory 3hr + Lab 2hr,B1,Dr. Kumar
```

**New Format:**
```csv
CourseCode,CourseType,BatchID,TeacherName
CO401,4,B1,Dr. Sharma
CO403,3+2,B1,Dr. Kumar
```

### Data Loader Fix

Modified `utils/data_loader.py` (lines 75-80) to parse both formats:

```python
# Support both old ("Theory 3hr + Lab 2hr") and new ("3+2") formats
if "Lab" in course_type_str or "+2" in course_type_str:
    self._courses[code].course_type = CourseType.THEORY_3HR_LAB_2HR
else:
    self._courses[code].course_type = CourseType.THEORY_4HR
```

### Room Configuration Update

Modified `agents/conflict_aware_planner.py` (lines 79-80):

```python
# Before
regular_rooms = [f"R{i}" for i in range(1, 24)]  # R1-R23

# After  
regular_rooms = [f"R{i}" for i in range(1, 29)]  # R1-R28
lab_rooms = [f"LAB{i}" for i in range(1, 8)]     # LAB1-LAB7 (unchanged)
```

---

## Phase 2: UCB Algorithm Explanation

### User Question
> "Explain UCB (Upper Confidence Bound)"

### Explanation Provided

**UCB Formula:**
```
score = mean_score + sqrt(2 * ln(total_trials) / trials_for_this_strategy)
        ↑              ↑
   EXPLOITATION    EXPLORATION
```

**Components:**
1. **Exploitation** (`mean_score`) - Use strategies that worked well before
2. **Exploration** (`sqrt(...)`) - Try strategies with fewer attempts

**In This Project:**
- SelectionAgent uses UCB to choose between: `greedy`, `random`, `best_of_n`, `conflict_aware`, `llm_guided`
- Untried algorithms get immediate priority
- Balances learning new approaches vs using proven ones

**Code Location:** `agents/selection_agent.py` method `_ucb_select()`

---

## Phase 3: Multi-Agent Trace Generation

### User Request
> "Generate step-by-step trace of the multi-agent workflow"

### Created: `run_with_trace.py`

New script that logs every agent action:

```
[ORCHESTRATOR] 🚀 Starting Multi-Agent Timetable Scheduling
    rooms_regular: 28
    rooms_lab: 7
[DATA_LOADER] 📚 Loading CSV data files
[DATA_LOADER] ✅ Data loaded - 158 courses, 59 teachers, 2525 students
[CONSTRAINT_AGENT] 🔒 Extracting constraints
[CONFLICT_AGENT] 🔍 Building student conflict matrix
    course_batches_with_conflicts: 235
    total_conflict_pairs: 9546
[SELECTION_AGENT] 🎯 UCB selects: conflict_aware
[PLANNER_AGENT] 📅 DSatur ordering + greedy slot assignment
    sessions_scheduled: 1064
    total_student_conflicts: 6458
[VERIFICATION_AGENT] ✅ Validation: 0 teacher, 0 room conflicts
```

### Output Files Generated
- `output/multi_agent_trace.json` - Machine-readable
- `output/multi_agent_trace.md` - Human-readable with ASCII diagrams

### Workflow Diagram Created

```
┌─────────────────┐
│   ORCHESTRATOR  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌──────────┐
│DATA   │  │CONSTRAINT│
│LOADER │  │  AGENT   │
└───┬───┘  └────┬─────┘
    └─────┬─────┘
          ▼
   ┌────────────┐
   │  CONFLICT  │
   │   AGENT    │
   └─────┬──────┘
         ▼
   ┌────────────┐
   │ SELECTION  │  ← UCB Algorithm
   │   AGENT    │
   └─────┬──────┘
         ▼
   ┌────────────┐
   │  PLANNER   │  ← DSatur + Greedy
   │   AGENT    │
   └─────┬──────┘
         ▼
   ┌────────────┐
   │VERIFICATION│
   │   AGENT    │
   └─────┬──────┘
         ▼
   ┌────────────┐
   │   MEMORY   │
   │   AGENT    │
   └────────────┘
```

---

## Phase 4: Student Conflict Analysis

### User Question
> "What is the root cause of the 6,458 student conflicts?"

### Analysis Conducted

**Student Distribution:**
```
📊 Students by courses enrolled:
   1 course:    3 students 
   4 courses:  154 students ███████
   5 courses:  756 students █████████████████████████████████████
   6 courses: 1176 students ██████████████████████████████████████████████████
   7 courses:  370 students ██████████████████
   8 courses:   44 students ██
   9 courses:    1 student 

Average: 5.7 courses per student
```

**Top Conflict Pairs:**
| Course Pair | Shared Students |
|-------------|-----------------|
| EC401-B1 ↔ EC403-B1 | 82 |
| EC401-B2 ↔ EC403-B2 | 82 |
| CE401-B1 ↔ CE403-B1 | 79 |
| CO401-B1 ↔ CO403-B1 | 75 |

**Root Cause:** Students in the same batch take the SAME set of courses together. If EC401-B1 and EC403-B1 are scheduled at the same time, 82 students would need to be in two places at once.

**Mathematical Inevitability:**
- 235 course-batches to schedule
- 40 time slots available (5 days × 8 hours)
- Average 5.9 course-batches per slot
- Some overlap is unavoidable

### Verification: Teacher and Room Conflicts

```python
# Ran verification script
👨‍🏫 TEACHER CONFLICTS: ✅ None
🏫 ROOM CONFLICTS: ✅ None
📊 Room utilization: 76.0% (1064/1400 slots used)
```

---

## Phase 5: LLM Usage Analysis

### User Question
> "Which agents use the LLM?"

### Discovery: Minimal LLM Usage

Searched for `_call_llm` in codebase:

| Agent | Uses LLM? | What For |
|-------|-----------|----------|
| **PlannerAgent** | ✅ Yes (2 calls) | Strategy selection |
| **LLMPlannerAgent** | ✅ Yes (1 call) | Full scheduling |
| **ConstraintAgent** | ❌ No | Hardcoded rules |
| **VerificationAgent** | ❌ No | Rule-based counting |
| **SelectionAgent** | ❌ No | UCB math algorithm |
| **ConflictAwarePlanner** | ❌ No | DSatur + Greedy |

### LLM Call Log (`logs/llm_calls.json`)

```json
{
  "total_calls": 1,
  "calls": [{
    "agent": "PlannerAgent",
    "prompt": "Provide strategy for scheduling...",
    "response": {"approach": "labs_first", "reasoning": "..."},
    "duration_ms": 21916
  }]
}
```

### User Observation
> "So we are using LLM for only 1%?"

**Answer:** Yes! The LLM only provided 3 configuration flags:
- `approach: "labs_first"` → Controls scheduling order
- `morning_theory: false` → Theory in afternoon
- `spread_across_days: true` → Distribute sessions

All other work (1064 slot assignments) is done algorithmically.

---

## Phase 6: LLM Enhancement Implementation

### User Request
> "We are not planning and verifying constraints using the LLM. Do it."

### Implementation Plan Approved

1. **ConstraintAgent** - Add LLM for constraint discovery
2. **VerificationAgent** - Add LLM for feedback generation
3. **RefinementAgent** (NEW) - Add LLM for slot swap suggestions

### Changes Made

#### 1. ConstraintAgent Enhancement

**File:** `agents/constraint_agent.py`
**Added:** `analyze_with_llm()` method

```python
def analyze_with_llm(self, courses, teachers, config, conflict_pairs):
    prompt = f"""Analyze this scheduling data:
    - Courses: {len(courses)}
    - Teachers: {len(teachers)}
    - High-conflict pairs: {conflict_pairs[:10]}
    
    Return: bottleneck_teachers, critical_course_pairs, 
            recommended_priority_order, suggested_constraints
    """
    response = self._call_llm(prompt)
    return json.loads(response)
```

**LLM Output Example:**
```json
{
  "bottleneck_teachers": ["Dr. Ritu Kumar", "Dr. Pooja Singh"],
  "recommended_priority_order": [
    "Laboratory sessions",
    "Courses taught by bottleneck teachers",
    "High-conflict course pairs"
  ],
  "suggested_constraints": 5
}
```

#### 2. VerificationAgent Enhancement

**File:** `agents/verification_agent.py`
**Added:** `get_llm_feedback()`, `suggest_improvements_with_llm()`

```python
def get_llm_feedback(self, proposal, conflicts, student_conflicts, courses):
    prompt = f"""Analyze scheduling results:
    - Coverage: {coverage}%
    - Student conflicts: {student_conflicts}
    
    Return: overall_assessment, root_causes, priority_actions
    """
```

**LLM Output Example:**
```json
{
  "overall_assessment": "critical",
  "root_causes": [
    "Severe resource bottleneck for 4th-year courses",
    "Lab room unavailability or fragmentation"
  ],
  "priority_actions": [
    "Audit and relax hard constraints",
    "Investigate IT431 resources"
  ]
}
```

#### 3. RefinementAgent (NEW)

**File:** `agents/refinement_agent.py` (created)

```python
class RefinementAgent(BaseAgent):
    def suggest_refinements(self, high_conflict_slots, low_conflict_slots):
        # LLM suggests slot swaps
        
    def apply_refinements(self, proposal, moves, pair_student_count):
        # Actually applies the swaps!
        
    def iterative_refinement(self, proposal, pair_student_count, max_iterations=3):
        # Loops until conflicts stop improving
```

#### 4. Unified Runner

**File:** `run_with_llm.py` (created)

Shows full LLM integration with 8 calls across 4 agents.

---

## Phase 7: LLM Action Loop Implementation

### User Question
> "How are you parsing these actionable insights and taking action?"

### Honest Assessment

| Agent | LLM Output | Actually Used? |
|-------|------------|----------------|
| PlannerAgent | `approach: "labs_first"` | ✅ Yes |
| ConstraintAgent | Bottleneck teachers | ❌ Logged only |
| VerificationAgent | Root causes | ❌ Logged only |
| RefinementAgent | Slot swaps | ❌ Logged only |

**Only ~20% of LLM insights were being acted upon!**

### User Request
> "Make LLM suggestions actually modify the schedule"

### Implementation: Full Action Loop

Rewrote `RefinementAgent.apply_refinements()` to:

1. **Parse LLM moves:**
```python
# LLM returns:
{"moves": [{"target": "HU325-B1", "from": "Friday 15", "to": "Thursday 12"}]}

# Parse target course-batch
parts = target.rsplit("-", 1)  # ["HU325", "B1"]
course, batch = parts[0], parts[1]

# Parse destination slot
to_parts = to_slot.split()  # ["Thursday", "12"]
to_day = Day(to_parts[0])
to_hour = int(to_parts[1])
```

2. **Validate constraints:**
```python
# Check teacher availability
if entry.teacher_name in teacher_schedule[new_slot_key]:
    self.log(f"Cannot move {target}: teacher busy")
    continue

# Check room availability
if entry.room_id in room_schedule[new_slot_key]:
    # Try to find alternative room
    alt_room = find_free_room(new_slot_key)
```

3. **Apply the move:**
```python
new_entry = ScheduleEntry(
    course_code=entry.course_code,
    batch_id=entry.batch_id,
    teacher_name=entry.teacher_name,
    room_id=room_to_use,
    time_slot=TimeSlot(day=to_day, hour=to_hour),
    session_type=entry.session_type,
    student_count=entry.student_count
)
entries[i] = new_entry
self.log(f"✅ Moved {target} from {old_slot} to {to_slot}")
```

### Iterative Refinement Results

```
[RefinementAgent] Starting iterative refinement. Initial conflicts: 6458

Iteration 1: 6458 → 6550 conflicts
  ✅ Moved HU325-B1: Friday 15:00 → Thursday 12:00
  ✅ Moved IT325-B1: Monday 10:00 → Thursday 10:00
  ✅ Moved CH307-B1: Monday 13:00 → Friday 17:00
  ❌ EC401-B1: teacher busy (blocked)

Iteration 2: 6550 → 6545 conflicts
  ✅ Moved IT325-B1: Thursday 10:00 → Friday 16:00
  ✅ Moved HU325-B1: Thursday 12:00 → Thursday 13:00

Iteration 3: 6545 → 6573 conflicts
  ✅ Moved IT321-B1: Monday 10:00 → Thursday 12:00
  ✅ Moved HU325-B1: Thursday 13:00 → Friday 16:00

Final: 6458 → 6573 (7 moves applied, slight increase)
```

### Key Finding

LLM suggestions **sometimes increase conflicts** because it doesn't have full visibility into which courses share students. Future improvement: pass conflict matrix data to LLM prompt.

---

## Final State & Files

### Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `utils/data_loader.py` | 75-80 | New CourseType format support |
| `agents/conflict_aware_planner.py` | 79-80 | Room config 28+7 |
| `agents/constraint_agent.py` | +80 lines | `analyze_with_llm()` |
| `agents/verification_agent.py` | +110 lines | LLM feedback methods |
| `agents/__init__.py` | +2 lines | Export RefinementAgent |

### Files Created

| File | Lines | Description |
|------|-------|-------------|
| `agents/refinement_agent.py` | 280 | LLM-guided slot swaps |
| `run_with_trace.py` | 406 | Algorithmic trace (0 LLM) |
| `run_with_llm.py` | 305 | Full LLM integration (8 calls) |
| `.gitignore` | 60 | Python/LLM patterns |
| `context.md` | 450+ | This file |

### Git Commits

1. `b0137a2` - Initial commit: Multi-agent timetable scheduler with LLM integration
2. `73310da` - Add context.md: Comprehensive development history

---

## Technical Reference

### Running the Schedulers

```bash
# Pure algorithmic (fast, 0 LLM calls, ~0.3 seconds)
python run_with_trace.py

# Full LLM integration (8 calls, ~140 seconds)
python run_with_llm.py

# Original CLI
python main.py --algorithm conflict_aware --rooms 35
```

### Environment Setup

```bash
# Required
pip install google-genai python-dotenv pandas

# .env file
GEMINI_API_KEY=your_api_key_here
```

### Key Data Files

| File | Records | Description |
|------|---------|-------------|
| `course_batch_teachers.csv` | 237 | Course-batch-teacher mapping |
| `student_allocations_aggregated.csv` | 2525 | Student enrollments |
| `course_batches.csv` | ~800 | Batch details |

### Performance Comparison

| Mode | LLM Calls | Time | Coverage | Conflicts |
|------|-----------|------|----------|-----------|
| `run_with_trace.py` | 0 | 0.29s | 100% | 6,458 |
| `run_with_llm.py` | 8 | 138s | 100% | 6,573 |

---

## Appendix: Full LLM Prompts

### ConstraintAgent Prompt

```
You are a scheduling constraint expert. Analyze this university timetabling problem.

PROBLEM DATA:
- Courses: 158
- Teachers: 59
- Total sessions: 1064
- Available slots: 1400 (71.4% utilization)

BUSIEST TEACHERS:
- Dr. Ritu Kumar: 8 courses
- Dr. Pooja Singh: 7 courses

HIGH-CONFLICT PAIRS:
- EC401-B1 and EC403-B1: 82 shared students

Return JSON with:
- bottleneck_teachers
- critical_course_pairs
- recommended_priority_order
- potential_issues
- suggested_constraints
```

### VerificationAgent Prompt

```
Analyze these scheduling results and provide feedback.

SCHEDULE RESULTS:
- Sessions scheduled: 1064/1064 (100% coverage)
- Student conflicts: 6458
- Hard constraint violations: 0

Return JSON with:
- overall_assessment: good|acceptable|needs_improvement|critical
- root_causes: [list]
- specific_fixes: [{issue, fix}]
- priority_actions: [top 3]
- estimated_improvement: string
```

### RefinementAgent Prompt

```
Suggest slot swaps to reduce student conflicts.

HIGH-CONFLICT SLOTS:
- Monday 10:00: 28 courses, 438 conflicts
- Monday 11:00: 27 courses, 412 conflicts

LOW-CONFLICT SLOTS:
- Friday 16:00: 18 courses, 23 conflicts
- Friday 17:00: 12 courses, 15 conflicts

Return JSON with:
- strategy: brief description
- moves: [{action, target, from, to, reason}]
- expected_reduction: estimate
- risks: [potential issues]
```

---

*End of Development Context*
