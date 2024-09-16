# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack, Queue, PriorityQueue
from game import Directions
from game import Actions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def constructPath(map, start, goal):
#     """
#     Returns the path of coordinates using map 
#     """
#     path = []
#     current = goal

#     while current != start:
#         path.append(current)
#         current = map[current]

#     path.append(start)
#     path.reverse()
#     return path

# def constructDirections(path):
#     """
#     Returns NESW directions from coordinate path 
#     """
#     if len(path) < 2:
#         return []
    
#     directions = []

#     for i in range(1, len(path)):
#         print(path)
#         curr = path[i]
#         prev = path[i-1]
#         print(type(curr))
#         print(curr)
#         vector = (curr[0] - prev[0], curr[1] - prev[1])
#         directions.append(Actions.vectorToDirection(vector))
    
#     return directions

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    start_state = problem.getStartState()
    
    if problem.isGoalState(start_state):
        return []

    stack = util.Stack()
    visited = []
    stack.push((start_state, []))

    while not stack.isEmpty():
        parent, actions = stack.pop()
        
        if parent not in visited:
            visited.append(parent)

            if problem.isGoalState(parent):
                return actions
            
            children = problem.getSuccessors(parent)

            for child in children:
                stack.push((child[0], actions + [child[1]]))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []

    queue = Queue()
    visited = []
    queue.push((start_state, []))

    while not queue.isEmpty():
        parent, actions = queue.pop()

        if parent not in visited:
            visited.append(parent)

            if problem.isGoalState(parent):
                return actions

            children = problem.getSuccessors(parent)

            for child in children:
                queue.push((child[0], actions + [child[1]]))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
        
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []

    visited = []
    queue = PriorityQueue()
    queue.push((start_state, [], 0), 0)

    while not queue.isEmpty():
        parent, actions, parent_cost = queue.pop()
        if parent not in visited:
            visited.append(parent)

            if problem.isGoalState(parent):
                print(actions)
                return actions
            
            children = problem.getSuccessors(parent)

            for child in children:
                priority = parent_cost + child[2]
                tup = (child[0], child[1], priority)
                queue.push(tup, priority)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    pq = PriorityQueue()
    visited = set()
    cost_map = {}
    path_map = {}

    start_state = problem.getStartState()
    pq.push(start_state, 0)  # Push start state with a priority of f(n) = g(n) + h(n)
    cost_map[start_state] = 0

    while not pq.isEmpty():
        parent = pq.pop()
        parent_cost = cost_map[parent]  # g(n), the actual cost to reach this node

        if parent in visited:
            continue

        if problem.isGoalState(parent):
            return constructDirections(constructPath(path_map, start_state, parent))

        visited.add(parent)

        children = problem.getSuccessors(parent)
        for child_full in children:
            child = child_full[0]
            child_cost = child_full[2]

            new_cost = parent_cost + child_cost  # g(n) + cost(child)
            heuristic_cost = heuristic(child, problem)  # h(n), the heuristic estimate
            total_cost = new_cost + heuristic_cost  # f(n) = g(n) + h(n)

            if child not in visited:
                if child not in cost_map or new_cost < cost_map[child]:
                    cost_map[child] = new_cost
                    path_map[child] = parent
                    pq.update(child, total_cost)  # Priority is based on f(n) = g(n) + h(n)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
