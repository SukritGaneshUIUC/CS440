# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush
import queue

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None.
    """

    # NOTE: POINTS ARE IN FORMAT: (alpha, beta, gamma)
    # beta and gamma are optional

    start = maze.getStart()[:maze.getAngleCount()]
    print('The Start:', start)
    objectives = maze.getObjectives()

    visited = set()                     # contains points (not Nodes) you've come across
    frontierQueue = queue.Queue()       # frontier queue
    frontierQueue.put(start)            # contains the list of Nodes
    parentDictionary = {}               # parent dictionary (for backtracing)

    while (frontierQueue.qsize() > 0):
        # print(frontierQueue.qsize())
        currentPoint = frontierQueue.get()
        visited.add(currentPoint)

        if (currentPoint in objectives):
            # backtrace
            print('Will Backtrace')
            return backtrace(currentPoint, parentDictionary)

        currentNeighbors = maze.getNeighbors(currentPoint)
        # print('cr:', currentNeighbors)
        for n in currentNeighbors:
            if (n not in visited):
                visited.add(n)
                frontierQueue.put(n)
                parentDictionary[n] = currentPoint

    return None

# Helper methods

def backtrace(currentPoint, parentDictionary):
    thePath = [currentPoint]
    while (currentPoint in parentDictionary):
        parentPoint = parentDictionary[currentPoint]
        thePath = [parentPoint] + thePath
        currentPoint = parentPoint
    return thePath

# Node class

class Node:

    def __init__(self, point):
        self.point = point

    def __lt__(self, other):
        return self.point < other.getPoint()

    def getPoint(self):
        return self.point

    def setPoint(self, newPoint):
        self.point = newPoint
