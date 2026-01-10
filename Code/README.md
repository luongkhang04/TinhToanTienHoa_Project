# Path Planning in Dynamic Environments

A path planning system that combines global and local planning algorithms to navigate through dynamic environments with moving obstacles.

## Features

- **Hierarchical Planning**: Combines global path planning with local reactive navigation
- **Multiple Algorithms**: Support for various planning algorithms at both levels
- **Dynamic Obstacles**: Handles environments with moving obstacles
- **Visualization**: Animated visualization of path execution
- **Map Support**: Load custom maps from PNG files

## Implemented Algorithms

### Global Planning Algorithms

Global planners compute high-level waypoints from start to goal using the static map:

1. **Grid BFS** (`grid_bfs`) - Breadth-First Search on grid
2. **Grid DFS** (`grid_dfs`) - Depth-First Search on grid  
3. **A*** (`astar`) - A* search with heuristic guidance
4. **Dijkstra** (`dijkstra`) - Dijkstra's shortest path algorithm

### Local Planning Algorithms

Local planners handle reactive navigation within the agent's observation range:

1. **Reactive BFS** (`reactive_bfs`) - BFS-based reactive planning
2. **Reactive DFS** (`reactive_dfs`) - DFS-based reactive planning
3. **Potential Field** (`potential_field`) - Artificial potential field method
4. **Greedy Local** (`greedy`) - Greedy best-first approach

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run with default settings (BFS global + reactive BFS local, 100 moving obstacles):
```bash
python main.py
```

### Custom Configuration

```bash
python main.py -m <map_file> -g <global_planner> -l <local_planner> -o <num_obstacles> -v
```

**Arguments:**
- `-m, --map`: Map file name (default: `1.png`)
  - Available maps: `1.png`, `2.png`, `3.png`, `4.png`, `5.png`, `6.png`, `7.png`
- `-g, --global`: Global planning algorithm (default: `grid_bfs`)
  - Choices: `grid_bfs`, `grid_dfs`, `astar`, `dijkstra`
- `-l, --local`: Local planning algorithm (default: `reactive_bfs`)
  - Choices: `reactive_bfs`, `reactive_dfs`, `potential_field`, `greedy`
- `-o, --obstacles`: Number of moving obstacles (default: `100`)
- `-v, --visualize`: Show animated visualization

### Examples

Plan with A* global and potential field local planner:
```bash
python main.py -g astar -l potential_field -v
```

Use map 2 with 200 moving obstacles:
```bash
python main.py -m 2.png -o 200 -v
```

Compare Dijkstra with greedy local planning:
```bash
python main.py -g dijkstra -l greedy -o 150 -v
```

## Project Structure

```
code2/
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
├── map/                    # Map files (PNG format)
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── planning/
│   ├── global_planning.py  # Global path planning algorithms
│   └── local_planning.py   # Local reactive planning algorithms
└── utils/
    ├── agent_graph.py      # Agent observation graph
    ├── DiscreteWorld.py    # Discrete world representation
    ├── Global.py           # Global environment manager
    ├── graph_utils.py      # Graph utilities
    ├── MDP.py              # MDP base class
    ├── move.py             # Movement utilities
    ├── Obstacle.py         # Obstacle and agent classes
    └── RandomMap.py        # Random map generation
```

## How It Works

1. **Global Planning**: Computes a high-level path from start to goal using the static map
2. **Local Planning**: Agent follows global waypoints while reactively avoiding moving obstacles within its observation range
3. **Obstacle Handling**: Moving obstacles with random motion patterns create dynamic challenges
4. **Replanning**: Local planner continuously adapts to new observations

## Map Format

Maps should be PNG files where:
- White pixels (value 1) = Free space
- Black pixels (value 0) = Static obstacles

Place custom maps in the `map/` directory.

## Output

The program outputs:
- Planning time for global path
- Number of waypoints in global path
- Execution details (steps taken, path length)
- Optional: Animated visualization showing agent path, observation range, and moving obstacles
