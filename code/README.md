# Mobile Robot Path Planning: Improved A* + DWA

This small demo implements the algorithm described in the paper "A Mobile Robot Path Planning Algorithm Based on Improved A* Algorithm and Dynamic Window Approach".

What is included
- `src/astar.py` — grid-based A* with an optional path smoothing step (improved A* behaviour).
- `src/dwa.py` — a simple Dynamic Window Approach local planner for differential-drive robots.
- `src/utils.py` — helpers for grids, collision checking, and distance maps.
- `examples/run_demo.py` — runnable demo that plans a global path with A* then follows it using DWA; shows a matplotlib animation.
- `requirements.txt` — minimal dependencies.

Quick start
1. (Optional) Create and activate a virtual environment.
2. Install requirements:

```powershell
python -m pip install -r requirements.txt
```

3. Run the demo:

```powershell
python examples\run_demo.py
```

Notes
- The demo uses a simple grid world and is intended as a clear, runnable reference implementation of the method in the paper (not a production-ready robot stack).