# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:06:34 2016

@author: hossam
"""
import random
import numpy
import time
import math
#import shift_scheduling_sat
import sklearn
from numpy import asarray
from sklearn.preprocessing import normalize
from sol import solution
from ElementSwitch import makeSwitch
from ElementSwitch import shiftSwitch



def normr(Mat):
    """normalize the columns of the matrix
    B= normr(A) normalizes the row
    the dtype of A is float"""
    Mat = Mat.reshape(1, -1)
    # Enforce dtype float
    if Mat.dtype != "float":
        Mat = asarray(Mat, dtype=float)

    # if statement to enforce dtype float
    B = normalize(Mat, norm="l2", axis=1)
    B = numpy.reshape(B, -1)
    return B


def randk(t):
    if (t % 2) == 0:
        s = 0.25
    else:
        s = 0.75
    return s


def RouletteWheelSelection(weights):
    accumulation = numpy.cumsum(weights)
    p = random.random() * accumulation[-1]
    chosen_index = -1
    for index in range(0, len(accumulation)):
        if accumulation[index] > p:
            chosen_index = index
            break

    choice = chosen_index

    return choice



def MVO(initial_solutions, objf, lb, ub, Max_time, printer, mvoScore_x_iterations):


    "parameters"
    # dim=30
    # lb=-100
    # ub=100
    dim = len(initial_solutions[0])
    # N = len(initial_solutions)


    WEP_Max = 1
    WEP_Min = 0.2
    # Max_time=1000
    N = len(initial_solutions)
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    #initializes array of universes
    Universes = numpy.zeros((N, dim))
    for i in range(N):
        Universes[i, :] = numpy.array(initial_solutions[i])

    Sorted_universes = numpy.copy(Universes)

    convergence = numpy.zeros(Max_time)

    #initializes best universe variable
    Best_universe = [0] * dim
    Best_universe_Inflation_rate = float("inf")

    s = solution()

    Time = 1
    ############################################
    print('MVO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    while Time < Max_time + 1:


        "Eq. (3.3) in the paper"
        WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / Max_time)

        TDR = 1 - (math.pow(Time, 1 / 6) / math.pow(Max_time, 1 / 6))

        #Initializes array of inflation rates
        Inflation_rates = [0] * len(Universes)

        for i in range(0, N):
            for j in range(dim):
                #Check if universes leave search space and bring them back 
                Universes[i, j] = numpy.clip(Universes[i, j], lb[j], ub[j])

            #Evaluate fitness 
            Inflation_rates[i] = objf(Universes[i, :])

            #If this fitness is lower than the best measured, this is now the best measured
            if Inflation_rates[i] < Best_universe_Inflation_rate:

                Best_universe_Inflation_rate = Inflation_rates[i]
                Best_universe = numpy.array(Universes[i, :])

        
        sorted_Inflation_rates = numpy.sort(Inflation_rates)
        sorted_indexes = numpy.argsort(Inflation_rates)

        #Re-sort universes by fitness
        for newindex in range(0, N):
            Sorted_universes[newindex, :] = numpy.array(
                Universes[sorted_indexes[newindex], :]
            )

        normalized_sorted_Inflation_rates = numpy.copy(normr(sorted_Inflation_rates))
        

        #Universes are sorted
        Universes[0, :] = numpy.array(Sorted_universes[0, :])

        positiveImpact = []
        # [impact, sendingSchedule, shift]
        bestImpact = []
        bestImpact.append(Sorted_universes[0])
        bestImpact.append(0)
        
        bestInflation = sorted_Inflation_rates[0]

        for i in range(1, N):
            
            Black_hole_index = i

            fitnessDif = []

            #Universes swap dimensional properties according to inflation rate
            # Try: swapping whole schedule
            for j in range(0, 28):
                r1 = float(random.randrange(int(normalized_sorted_Inflation_rates[0] * 100000000), int(normalized_sorted_Inflation_rates[-1] * 100000000)) / 100000000)
                
                if r1 < normalized_sorted_Inflation_rates[i]:
                    
                    White_hole_index = RouletteWheelSelection(-sorted_Inflation_rates)

                    if White_hole_index == -1:
                        White_hole_index = 0
                    #White_hole_index = 0

                    
                    savedUni = []
                    for k in range(dim):
                        savedUni.append(Universes[Black_hole_index, k])

                    fitness = objf(Universes[Black_hole_index])
                    shiftSwitch(j, Universes[Black_hole_index], Sorted_universes[White_hole_index], objf)
                    #makeSwitch(Universes[Black_hole_index], Sorted_universes[White_hole_index], j, objf)
                    after = objf(Universes[Black_hole_index])
                    dif = fitness - after

                    if dif < 0:
                        Universes[Black_hole_index] = savedUni
                    if after < bestInflation:
                        bestInflation = after 
                        bestImpact[0] = Universes[Black_hole_index]
                        bestImpact[1] = j
                        Best_universe = Universes[Black_hole_index]
                        
                    #     for h in range(1, N):
                    #         fitness = objf(Universes[h])
                    #         savedUni = []
                    #         for k in range(len(Universes[h])):
                    #             savedUni.append(Universes[h, k])
                    #         shiftSwitch(bestImpact[1], Universes[h], bestImpact[0])
                    #         after = objf(Universes[h])
                    #         dif = fitness - after
                    #         if dif < 0:
                    #             Universes[h] = savedUni
                    #         if after < bestInflation:
                    #             bestInflation = after
                    #             bestImpact[0] = Universes[h]
                    #             Best_universe = Universes[h]
                            
                        

                        
                r2 = random.random()
                
                #Worm holes appeear to randomly distribute dimensions from best universe
                if r2 < WEP:

                    savedUni = []
                    for k in range(len(Universes[Black_hole_index])):
                        savedUni.append(Universes[Black_hole_index, k])
                    fitness = objf(Universes[Black_hole_index])
                    # makeSwitch(Universes[Black_hole_index], Best_universe, j, objf)                    
                    shiftSwitch(j, Universes[Black_hole_index], Best_universe, objf)
                    after = objf(Universes[Black_hole_index])
                    dif = fitness - after
                    if dif < 0:
                        Universes[Black_hole_index] = savedUni
                    if after < bestInflation:
                        bestInflation = after
                        bestImpact[0] = Best_universe
                        bestImpact[1] = j

                        # for h in range(1, N):
                        #     fitness = objf(Universes[h])
                        #     savedUni = []
                        #     for k in range(len(Universes[h])):
                        #         savedUni.append(Universes[h, k])
                        #     shiftSwitch(bestImpact[1], Universes[h], bestImpact[0], objf)
                        #     after = objf(Universes[h])
                        #     dif = fitness - after
                        #     if dif < 0:
                        #         Universes[h] = savedUni

        for h in range(1, N):
            fitness = objf(Universes[h])
            savedUni = []
            for k in range(len(Universes[h])):
                savedUni.append(Universes[h, k])
            shiftSwitch(bestImpact[1], Universes[h], bestImpact[0], objf)
            after = objf(Universes[h])
            dif = fitness - after
            if dif < 0:
                Universes[h] = savedUni


            
        for i in range(len(Universes)):
            print(objf(Universes[i]))
        
            


        convergence[Time - 1] = Best_universe_Inflation_rate
        if Time % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(Time)
                    + " the best fitness is "
                    + str(Best_universe_Inflation_rate)
                ]
            )
            mvoScore_x_iterations.append(Best_universe_Inflation_rate)



        Time = Time + 1

        #if Time == 2 or Time == 3:
            # for universe in Universes:
            #     print(Best_universe - universe)
            #     sum = 0
            #     for i in range(len(universe)):
            #         sum += universe[i]
            #     print(sum)
            # print("Best")
            # printer(Best_universe)
            # print("Others")
            # for uni in Universes:
            #     printer(uni)

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "MVO"
    s.objfname = objf.__name__

    return s
