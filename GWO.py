# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016
@author: Hossam Faris
"""

import random
import numpy
import math
from solution import solution
import time
import psutil
import os

def nurseNumber(index):
    shift = index % 28
    return (index - shift) / 28

def shiftNumber(index):
    return index % 28

def shiftType(index):
    return index % 2

def weekNumber(index):
    shift = index % 28
    if shift > 13:
        return 1
    else:
        return 0

def enhancedGreyWolf(Alpha_Pos, objf, alphasHunting, skipIndex):

    index = 0
    nurse = 0
    week = 0

    alphaFoundPrey = 0

    for pos in Alpha_Pos:
    

        if pos == 1 and shiftNumber(index) < 25:
            # print ("Index", index, "nurse number:", nurseNumber(index), "Week: ", weekNumber(index), 
            # "Shift:", shiftNumber(index),
            # "Shift Type:", shiftType(index), "Position:", pos)
            for swapNurse in range(0, (25 - skipIndex)):

                swapNurseIndex = (shiftNumber(index)) + swapNurse * 28
                swapNursePos = Alpha_Pos[swapNurseIndex]

                if (shiftNumber(index) > 0):
                    swapNurseIndex + 1
                # print(nurseNumber(index))
                # print(shiftNumber(index))
                # print(swapNurseIndex)

                if (index + skipIndex > 699):
                    skipIndex = 1

                if (nurseNumber(index) != nurseNumber(swapNurseIndex) 
                    and Alpha_Pos[index] == 1 and Alpha_Pos[index + skipIndex] == 0
                    and Alpha_Pos[swapNurseIndex + skipIndex] == 1 and Alpha_Pos[swapNurseIndex] == 0):
                    fitnessBefore = objf(Alpha_Pos)
                    # print("Before swap", fitnessBefore)
                    # print("Swapping nurse:", nurseNumber(index), "with nurse: ", nurseNumber(swapNurseIndex))
                    # printer(simulation)

                    # swap current index
                    Alpha_Pos[index] = 0
                    Alpha_Pos[index + skipIndex] = 1

                    Alpha_Pos[swapNurseIndex] = 1
                    Alpha_Pos[swapNurseIndex + skipIndex] = 0

                    fitnessAfter = objf(Alpha_Pos)

                    
                    # print("After swap", fitnessAfter)
                    if fitnessAfter <= fitnessBefore:
                        winner = fitnessAfter
                        # print(winner)
                        alphaFoundPrey += 1

                        if alphaFoundPrey == alphasHunting:
                            return Alpha_Pos
                        
                    else:
                        Alpha_Pos[index] = 1
                        Alpha_Pos[index + skipIndex] = 0
                        
                        Alpha_Pos[swapNurseIndex] = 0
                        Alpha_Pos[swapNurseIndex + skipIndex] = 1
                    # printer(simulation)
        index += 1
    return Alpha_Pos

def organizePack(Alpha_Pos, objf, alphasHunting):

    index = 0
    nurse = 0
    week = 0
    alphasHunting = 625
    alphaFoundPrey = 0

    for pos in Alpha_Pos:
    

        if shiftNumber(index) < 28:

            for swapNurse in range(0, 25):

                swapNurseIndex = (shiftNumber(index)) + swapNurse * 28

                if (shiftNumber(index) > 0):
                    swapNurseIndex + 1

                if (nurseNumber(index) != nurseNumber(swapNurseIndex)
                    and Alpha_Pos[index] != Alpha_Pos[swapNurseIndex]
                    and shiftNumber(index) == shiftNumber(swapNurseIndex)):
                    
                    fitnessBefore = objf(Alpha_Pos)
                    
                    # swap current index
                    indexPos = Alpha_Pos[index]
                    swapPos = Alpha_Pos[swapNurseIndex]
                    Alpha_Pos[index] = swapPos
                    Alpha_Pos[swapNurseIndex] = indexPos

                    fitnessAfter = objf(Alpha_Pos)
                    # print(fitnessAfter)
                    
                    # print("After swap", fitnessAfter)
                    if fitnessAfter <= fitnessBefore:
                        winner = fitnessAfter
                        alphaFoundPrey += 1

                        if alphaFoundPrey == alphasHunting:
                            return Alpha_Pos
                        
                    else:
                        Alpha_Pos[index] = indexPos
                        Alpha_Pos[swapNurseIndex] = swapPos



        index += 1
    return Alpha_Pos

def GWO(initial_solutions, objf, lb, ub, Max_iter, printer, 
gwoScore_x_iterations, gwoScore_x_time, gwoCPU_x_iterations, gwoRAM_x_iterations):

    # Max_iter=1000
    # lb=-100
    # ub=100
    # dim=30
    # SearchAgents_no=5
    winner = 99999999
    skipIndex = 1

    alphasHunting = 20
    stuckIndex = 0

    dim = len(initial_solutions[0])
    SearchAgents_no = len(initial_solutions)
    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        # Positions[:, i] = (
        #     numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        # )
        Positions[i, :] = numpy.array(initial_solutions[i])

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('GWO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])
            # print(fitness)
            # printer(Alpha_pos)
            # printer(Beta_pos)
            # printer(Delta_pos)

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                # Update alpha
                Alpha_pos = Positions[i, :].copy()
                for i in range(SearchAgents_no):
                    Positions[i, :] = numpy.array(Positions[i, :].copy())

            if fitness > Alpha_score and fitness < Beta_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / Max_iter)
        # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a
                # Equation (3.3)
                C1 = 2 * r2
                # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha
                # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                # Equation (3.3)
                C2 = 2 * r2
                # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta
                # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                # Equation (3.3)
                C3 = 2 * r2
                # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta
                # Equation (3.5)-part 3

                roundedPos = round((X1 + X2 + X3) / 3)

                if  roundedPos > 1:
                    Positions[i, j] = 1
                
                if  roundedPos < 0:
                    Positions[i, j] = 0

                # Positions[i] = bestPos
                

        Convergence_curve[l] = Alpha_score

        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(Alpha_score)]
            )
            gwoScore_x_iterations.append(Alpha_score)
            process = psutil.Process(os.getpid())
            # print(process.memory_info().vms/ 1024 / 1024)
            gwoRAM_x_iterations.append(psutil.virtual_memory()[2])
            gwoCPU_x_iterations.append(psutil.cpu_percent(1))
            # printer(bestPos)

            Alpha_pos = enhancedGreyWolf(Alpha_pos, objf, alphasHunting, skipIndex % 28)
            Alpha_score = objf(Alpha_pos)

            if (winner == Alpha_score):
                skipIndex += 1
                stuckIndex += 1

            if (stuckIndex == 5):
                print("Waiting wolf alphas re-organize the pack")
                Alpha_pos = organizePack(Alpha_pos, objf, alphasHunting)
                Alpha_score = objf(Alpha_pos)
                stuckIndex = 0

            
            winner = Alpha_score

            

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GWO"
    s.objfname = objf.__name__

    return s


