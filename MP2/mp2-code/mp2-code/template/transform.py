
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.

        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """

    # step 1: initialize maze matrix
    mm = initializeMazeMatrix(granularity, arm)

    offsets = getOffsets(arm)
    initialAngles = arm.getArmAngle()
    print(initialAngles)

    # step 2: fill out maze (P for start, . for goal, % for obstacle, ' ' for open space)

    # mark start`
    startIndices = [0, 0, 0]
    temps = angleToIdx(initialAngles, offsets, granularity)
    startIndices[:len(temps)] = list(temps)
    mm[startIndices[0]][startIndices[1]][startIndices[2]] = 'P'
    print('Start', startIndices, ':', mm[startIndices[0]][startIndices[1]][startIndices[2]])
    print('Dimensions:', len(mm), len(mm[0]), len(mm[0][0]))

    # mark spaces
    mm = markSpaces(arm, goals, obstacles, window, granularity, mm)

    # step 3: create maze object and return

    saveMazeToFile(mm)
    return Maze(mm, offsets, granularity)

def markSpaces(arm, goals, obstacles, window, granularity, mm):
    offsets = getOffsets(arm)

    g = 0
    o = 0
    bb = 0

    for a in range(len(mm)):
        print(a)
        for b in range(len(mm[0])):
            for c in range(len(mm[0][0])):
                currentAngles = idxToAngle((a,b,c)[:len(offsets)], offsets, granularity)
                arm.setArmAngle(currentAngles)
                # print(a, ' ', b, ' ', c, '|', len(mm), ' ', len(mm[0]), '', len(mm[0][0]))
                if (doesArmTipTouchGoals(arm.getEnd(), goals)):
                    mm[a][b][c] = '.'
                    g += 1
                if (doesArmTouchObjects(arm.getArmPosDist(), obstacles, isGoal=False)):
                    mm[a][b][c] = '%'
                    o += 1
                if (not isArmWithinWindow(arm.getArmPos(), window)):
                    mm[a][b][c] = '%'
                    bb += 1


    print('Goals:', g, 'Obstacles:', o, 'Borders:', bb)
    return mm

def initializeMazeMatrix(granularity, arm):
    armLimits = arm.getArmLimit()
    mm = []
    mmDimensions = [1, 1, 1]

    for i in range(len(armLimits)):
        linkLimit = armLimits[i]
        mmDimensions[i] = int((linkLimit[1] - linkLimit[0]) / granularity + 1)

    for i in range(mmDimensions[0]):
        currPlane = []
        for j in range(mmDimensions[1]):
            currRow = []
            for k in range(mmDimensions[2]):
                currRow.append(' ')
            currPlane.append(currRow)
        mm.append(currPlane)
    return mm

def getOffsets(arm):
    offsets = []
    armLimits = arm.getArmLimit()
    for al in armLimits:
        offsets.append(al[0])
    return tuple(offsets)

def saveMazeToFile(mm):
    f = open('tm.txt', 'w')
    for row in range(len(mm)):
        for col in range(len(mm[0])):
            f.write(mm[row][col][0])
        if (row < len(mm) - 1):
            f.write('\n')
    f.close()
