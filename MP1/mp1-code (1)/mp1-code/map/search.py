# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

# Import libraries
import copy
import heapq
import queue

# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfsSingle(maze, start, goal): # run BFS for finding a single point
    exploredPoints = []         # contains points you've fully explored
    frontierQueue = queue.Queue()
    frontierQueue.put(start)   # contains the list of (h, g, point)
    parentArray = initializeParentArray(maze)   # U, D, L, R, S for backtrace

    while (frontierQueue.qsize() > 0):
        currPt = frontierQueue.get()
        exploredPoints.append(currentPoint)

        # if your current point is the goal (meaning that it has the lowest cost AND it's on the goal)
        # you return
        if (currPt == goal):
            # backtrace
            return backtrace(parentArray, goal)

        currentNeighbors = maze.getNeighbors(currPt[0], currPt[1])
        for n in currentNeighbors:
            # add to frontier queue only if point isn't on frontier or explored AND if point is not wall
            if (n not in exploredPoints):  # n not in exploredPoints
                frontierQueue.put(n)
                parentArray = markParentArray(parentArray, currPt, n)

    return []

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    currPt = maze.getStart()
    path = []
    objectives = maze.getObjectives()

    for o in objectives:
        path += bfsSingle(maze, currPt, o)
        currPt = o

    return path

def initializeParentArray(maze):
    parentArray = []    # U, D, L, R, S for backtrace
    for i in range(maze.getDimensions()[0]):
        t = []
        for j in range(maze.getDimensions()[1]):
            t.append('')
        parentArray.append(t)
    parentArray[start[0]][start[1]] = 'S'
    return parentArray

def markParentArray(parentArray, currPt, n):
    if (currPt[1] > n[1]):
        parentArray[n[0]][n[1]] = 'U'
    elif (currPt[1] < n[1]):
        parentArray[n[0]][n[1]] = 'D'
    elif (currPt[0] < n[0]):
        parentArray[n[0]][n[1]] = 'L'
    elif (currPt[0] > n[0]):
        parentArray[n[0]][n[1]] = 'R'
    else:
        parentArray[n[0]][n[1]] = 'S'

    return parentArray

def backtrace(parentArray, goal):
    currPt = goal
    backtraceArray = [goal]
    while (parentArray[currPt[0]][currPt[1]] != 'S'):
        if (parentArray[currPt[0]][currPt[1]] == 'U'):
            currPt = (currPt[0], currPt[1] + 1)
        elif (parentArray[currPt[0]][currPt[1]] == 'D'):
            currPt = (currPt[0], currPt[1] - 1)
        elif (parentArray[currPt[0]][currPt[1]] == 'L'):
            currPt = (currPt[0] - 1, currPt[1])
        elif (parentArray[currPt[0]][currPt[1]] == 'R'):
            currPt = (currPt[0] + 1, currPt[1])
        backtraceArray = [currPt] + backtraceArray

    return backtraceArray

def manhattanDistance(p1, p2):
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

def simpleHeuristic(point, goal):
    return manhattanDistance(point, goal)

# # calculate cost by summing path length and distance to goal
# def getLowestCostPoint(frontierQueue, goal):
#     lowestCost = frontierQueue[0][0]
#     lowestCostPoint = frontierQueue[0][2]
#     for currPoint in frontierQueue:
#         if (currPoint[0] < lowestCost):
#             lowestCost = currPoint[0]
#             lowestCostPoint = currPoint[1]
#
#     return lowestCostPoint

def astarSingle(maze, start, goal):
    exploredPoints = []         # contains points you've fully explored
    frontierQueue = queue.PriorityQueue()
    frontierQueue.put([0 + simpleHeuristic(start, goal), 0, start])   # contains the list of (h, g, point)
    parentArray = []    # U, D, L, R, S for backtrace
    for i in range(maze.getDimensions()[0]):
        t = []
        for j in range(maze.getDimensions()[1]):
            t.append('')
        parentArray.append(t)
    parentArray[start[0]][start[1]] = 'S'


    while (frontierQueue.qsize() > 0):
        currentPoint = frontierQueue.get()
        exploredPoints.append(currentPoint[2])

        # if your current point is the goal (meaning that it has the lowest cost AND it's on the goal)
        # you return
        if (currentPoint[2] == goal):
            # backtrace
            return backtrace(parentArray, goal)

        currentNeighbors = maze.getNeighbors(currentPoint[2][0], currentPoint[2][1])
        for n in currentNeighbors:
            # add to frontier queue only if point isn't on frontier or explored AND if point is not wall
            if (n not in exploredPoints):  # n not in exploredPoints
                frontierQueue.put([currentPoint[1] + simpleHeuristic(n, goal), currentPoint[1] + 1, n])
                parentArray = markParentArray(parentArray, currentPoint[2], n)

    return []

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # currentPoint = maze.getStart()
    # path = []
    # objectives = maze.getObjectives()
    #
    # for o in objectives:
    #     path += astarSingle(maze, currentPoint, o)
    #     currentPoint = o
    #
    # return path
    return []

############################################### CORNER ###########################################

# distance to nearest objective x, summed with all distances from x to other objectives
def cornerHeuristic(point, objectives):
    # find objective closest to point
    objList = []
    for o in objectives:
        heapq.heappush(objList, [manhattanDistance(point, o), o])
    closestObjective = objList[0][1]
    h = heapq.heappop(objList)[0]   # contains distance from point to closest objective

    # # calculate distances to all other objectives (and add it to h)
    # for o in objectives:
    #     h += manhattanDistance(closestObjective, o)

    return h

def astarCornerSingle(maze, start, objectives):
    exploredPoints = []         # contains points you've fully explored
    frontierQueue = queue.PriorityQueue()
    frontierQueue.put([0 + cornerHeuristic(start, objectives), 0, start])   # contains the list of (h, g, point)
    parentArray = []    # U, D, L, R, S for backtrace
    for i in range(maze.getDimensions()[0]):
        t = []
        for j in range(maze.getDimensions()[1]):
            t.append('')
        parentArray.append(t)
    parentArray[start[0]][start[1]] = 'S'


    while (frontierQueue.qsize() > 0):
        currentPoint = frontierQueue.get()
        exploredPoints.append(currentPoint[2])

        # if your current point is the goal (meaning that it has the lowest cost AND it's on the goal)
        # you return
        if (currentPoint[2] in objectives):
            objectives.remove(currentPoint[2])
            # backtrace
            return backtrace(parentArray, currentPoint[2]), objectives, currentPoint[2]

        currentNeighbors = maze.getNeighbors(currentPoint[2][0], currentPoint[2][1])
        for n in currentNeighbors:
            # add to frontier queue only if point isn't on frontier or explored AND if point is not wall
            if (n not in exploredPoints):  # n not in exploredPoints
                frontierQueue.put([currentPoint[1] + cornerHeuristic(n, objectives), currentPoint[1] + 1, n])
                parentArray = markParentArray(parentArray, currentPoint[2], n)

    return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    # currentPoint = maze.getStart()
    # path = []
    # objectives = maze.getObjectives()
    #
    # # Use Kruskal's to order objectives
    # orderedObjectives = []
    #
    # while(len(objectives) > 0):
    #     currentPath, objectives, currentPoint = astarCornerSingle(maze, currentPoint, objectives)
    #     path += currentPath
    #
    # return path
    return []

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []

# GRAPH CLASS

#Class to represent a graph
class Graph:

    def __init__(self,vertices):
        self.V= vertices #No. of vertices
        self.graph = [] # default dictionary
                                # to store graph


    # function to add an edge to graph
    def addEdge(self,u,v,w):
        self.graph.append([u,v,w])

    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else :
            parent[yroot] = xroot
            rank[xroot] += 1

# The main function to construct MST using Kruskal's algorithm
# Credit to Geeks4Geeks
def KruskalMST(self):

	result =[] #This will store the resultant MST

	i = 0 # An index variable, used for sorted edges
	e = 0 # An index variable, used for result[]

		# Step 1: Sort all the edges in non-decreasing
			# order of their
			# weight. If we are not allowed to change the
			# given graph, we can create a copy of graph
	self.graph = sorted(self.graph,key=lambda item: item[2])

	parent = [] ; rank = []

	# Create V subsets with single elements
	for node in range(self.V):
		parent.append(node)
		rank.append(0)

	# Number of edges to be taken is equal to V-1
	while e < self.V -1 :

		# Step 2: Pick the smallest edge and increment
				# the index for next iteration
		u,v,w = self.graph[i]
		i = i + 1
		x = self.find(parent, u)
		y = self.find(parent ,v)

		# If including this edge does't cause cycle,
					# include it in result and increment the index
					# of result for next edge
		if x != y:
			e = e + 1
			result.append([u,v,w])
			self.union(parent, rank, x, y)
		# Else discard the edge

	# print the contents of result[] to display the built MST
	print("Following are the edges in the constructed MST")
	for u,v,weight in result:
		#print str(u) + " -- " + str(v) + " == " + str(weight)
		print ("%d -- %d == %d" % (u,v,weight))
