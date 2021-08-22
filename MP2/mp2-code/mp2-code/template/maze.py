# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) and
#            Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018
"""
This file contains the Maze class, which reads in a maze file and creates
a representation of the maze that is exposed through a simple interface.
"""

import copy
from const import *
from util import *

class Maze:
    # Initializes the Maze object by reading the maze from a file
    def __init__(self, input_map, offsets, granularity):
        self.__start = None
        self.__objective = []

        self.offsets = offsets
        self.angleCount = len(offsets)
        self.granularity = granularity

        self.__dimensions = [len(input_map), len(input_map[0]), len(input_map[0][0])]
        self.__map = input_map
        for x in range(self.__dimensions[ALPHA]):
            for y in range(self.__dimensions[BETA]):
                for z in range(self.__dimensions[GAMMA]):
                    if self.__map[x][y][z] == START_CHAR:
                        st = [0, 0, 0]
                        st[:self.angleCount] = idxToAngle((x, y, z)[:self.angleCount], self.offsets, granularity)
                        self.__start = tuple(st)
                        # self.__start = idxToAngle((x, y, z)[:self.angleCount], self.offsets, granularity)
                    elif self.__map[x][y][z] == OBJECTIVE_CHAR:
                        self.__objective.append(idxToAngle((x, y, z)[:self.angleCount], self.offsets, granularity))

        if not self.__start:
            print("Maze has no start")
            raise SystemExit

        if not self.__objective:
            print("Maze has no objectives")
            raise SystemExit

    def getChar(self, angles):
        indices = [0,0,0]
        indices[:self.angleCount] = list(angleToIdx(angles[:self.angleCount], self.offsets, self.granularity))
        return self.__map[indices[0]][indices[1]][indices[2]]

    # Returns True if the given position is the location of a wall
    def isWall(self, angles):
        return self.getChar(angles) == WALL_CHAR

    # Rturns True if the given position is the location of an objective
    def isObjective(self, angles):
        return self.getChar(angles) == OBJECTIVE_CHAR

    # Returns the start position as a tuple of (beta, column)
    def getStart(self):
        return self.__start

    def setStart(self, start):
        self.__start = start

    # Returns the dimensions of the maze as a (beta, column) tuple
    def getDimensions(self):
        return self.__dimensions

    # Returns the list of objective positions of the maze
    def getObjectives(self):
        return copy.deepcopy(self.__objective)

    def setObjectives(self, objectives):
        self.__objective = objectives

    def getAngleCount(self):
        return self.angleCount

    # Check if the agent can move into a specific beta and column
    def isValidMove(self, angles):
        indices = [0,0,0]
        indices[:self.angleCount] = list(angleToIdx((angles), self.offsets, self.granularity))
        return indices[0] >= 0 and indices[0] < self.getDimensions()[ALPHA] and \
               indices[1] >= 0 and indices[1] < self.getDimensions()[BETA] and \
               indices[2] >= 0 and indices[2] < self.getDimensions()[GAMMA] and \
               not self.isWall(angles)

    # Returns list of neighboing squares that can be moved to from the given alpha,beta,gamma
    def getNeighbors(self, angles):
        defaultAngles = [0,0,0]
        defaultAngles[:self.angleCount] = list(angles)
        defaultAngles = tuple(defaultAngles)
        alpha = defaultAngles[0]
        beta = defaultAngles[1]
        gamma = defaultAngles[2]

        possibleNeighbors = [
            (alpha + self.granularity, beta, gamma),
            (alpha - self.granularity, beta, gamma),
            (alpha, beta + self.granularity, gamma),
            (alpha, beta - self.granularity, gamma),
            (alpha, beta, gamma + self.granularity),
            (alpha, beta, gamma - self.granularity)
        ]
        neighbors = []
        for angles in possibleNeighbors:
            if self.isValidMove(angles[:self.angleCount]):
                neighbors.append(angles[:self.angleCount])

                # cn = [0, 0, 0]
                # cn[:self.angleCount] = angles[:self.angleCount]
                # cn = tuple(cn)
                # neighbors.append(cn)

                # neighbors.append(angles)
        # print('cr:', neighbors)
        return neighbors

    def saveToFile(self, filename):
        outputMap = ""
        for beta in range(self.__dimensions[1]):
            for alpha in range(self.__dimensions[0]):
                outputMap += self.__map[alpha][beta]
            outputMap += "\n"

        with open(filename, 'w') as f:
            f.write(outputMap)

        return True


    def isValidPath(self, path):
        # First, check whether it moves single hop
        for i in range(1, len(path)):
            prev = path[i-1]
            cur = path[i]
            dist = abs(prev[0]-cur[0]) + abs(prev[1]-cur[1]) +  abs(prev[2]-cur[2])
            if dist != self.granularity:
                return "Not single hop"

        # Second, check whether it is valid move
        for pos in path:
            if not self.isValidMove(pos[0], pos[1], pos[2]):
                return "Not valid move"


        # Last, check whether it ends up at one of goals
        if not path[-1] in self.__objective:
            return "Last position is not a goal state"

        return "Valid"

    def get_map(self):
        return self.__map
