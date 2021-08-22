# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position (int,int):of the arm link, (x-coordinate, y-coordinate)
    """

    run = int(math.cos(angle * math.pi / 180) * length)
    rise = int(math.sin(angle * math.pi / 180) * length)

    rise *= -1  # "higher" points have lower y-value, so y-axis is technically inverted

    newPoint = (start[0] + run, start[1] + rise)

    return newPoint

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """

    for armLink in armPosDist:
        for object in objects:
            start = armLink[0]
            end = armLink[1]
            padding = armLink[2]
            objRadius = object[-1]
            objPos = object[:-1]

            # euclidean distance from start of arm link to object
            startDist = euclideanDistance(start, objPos)
            # euclidean distance from end of arm link to object
            endDist = euclideanDistance(end, objPos)
            # distance from arm "center" to point
            segDist = pointToSegmentDistance(objPos, start, end)

            if (isGoal):
                padding = 0

            # startCount = 0
            # endCount = 0
            # segCount = 0
            if (startDist <= padding + objRadius):
                # print(start, objPos, euclideanDistance(start, objPos), padding + objRadius)
                # print(objects)
                # print('Comp Start:', startDist, padding + objRadius)
                return True
            if (endDist <= padding + objRadius):
                # print('Comp End:', endDist, padding + objRadius)
                return True
            if (segDist <= padding + objRadius):
                # print('Comp Seg:', segDist, padding + objRadius)
                return True

    return False

def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tick touch goals

        Args:
            armEnd (tuple): the arm tick position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tip touches any goal. False if not.
    """

    for goal in goals:
        if(euclideanDistance(armEnd, goal) <= goal[2]):
            return True

    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """

    for armLink in armPos:
        if ((withinWindow(armLink[0], window) == False) or (withinWindow(armLink[1], window) == False)):
            return False

    return True

def withinWindow(p, window):
    for i in range(len(window)):
        if (not (p[i] > 0 and p[i] <= window[i])):
            return False
    return True

def euclideanDistance(p1, p2):
    sumDiffSquare = 0
    for i in range(len(p1)):
        sumDiffSquare += (p2[i] - p1[i]) ** 2
    return sumDiffSquare ** 0.5

# # More assistance: https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
#
# def lineseg_dist(p, a, b):
#     p = np.array(p)
#     a = np.array(a)
#     b = np.array(b)
#
#     # normalized tangent vector
#     d = np.divide(b - a, np.linalg.norm(b - a))
#
#     # signed parallel distance components
#     s = np.dot(a - p, d)
#     t = np.dot(p - b, d)
#
#     # clamped parallel distance
#     h = np.maximum.reduce([s, t, 0])
#
#     # perpendicular distance component
#     c = np.cross(p - a, d)
#
#     return np.hypot(h, np.linalg.norm(c))

# assistance: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def pointToSegmentDistance(p, segStart, segEnd):
    baseVecDistSquare = euclideanDistance(segStart, segEnd) ** 2
    if (baseVecDistSquare == 0):
        return euclideanDistance(p, segStart)

    # calculate t
    t = 0
    for i in range(len(p)):
        t += (p[i] - segStart[i]) * (segEnd[i] - segStart[i])
    t = t / baseVecDistSquare

    t = max(0, min(1, t))   # normalize

    # calculate project point
    projectPoint = []
    for i in range(len(p)):
        projectPoint.append(segStart[i] + t * (segEnd[i] - segStart[i]))
    projectPoint = tuple(projectPoint)

    return euclideanDistance(p, projectPoint)

if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTickTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
