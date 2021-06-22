#!/usr/bin/env python3
# Copyright 2010-2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Creates a shift scheduling problem and solves it."""

from absl import app
from absl import flags

import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
from google.protobuf import text_format
import numpy as np

import MFO
import GWO
import MVO

FLAGS = flags.FLAGS

flags.DEFINE_string('output_proto', '',
					'Output file to write the cp_model proto to.')
flags.DEFINE_string('params', 'max_time_in_seconds:300.0',
					'Sat solver parameters.')
					
					
# Data
num_employees = 25
num_weeks = 2
shifts = ['D', 'N']
day_shift = 0
night_shift = 1
num_days = num_weeks * 7
num_shifts = len(shifts)

solutions_to_find = 30

mfoScore_x_iterations = []
gwoScore_x_iterations = []
mvoScore_x_iterations = []

mfoScore_x_time = []
gwoScore_x_time = []
mvoScore_x_time = []

mfoCPU_x_iterations = []
gwoCPU_x_iterations = []
mvoCPU_x_iterations = []

mfoRAM_x_iterations = []
gwoRAM_x_iterations = []
mvoRAM_x_iterations = []

interactions = 100
interactionsArray = np.arange(start=1, stop=interactions+1, step=1)
gwoIterations = []
mvoIterations = []

mfoScore_x_time = []
gwoScore_x_time = []
mvoScore_x_time = []

mfoTime = []
gwoTime = []
mvoTime = []

shift_penalties = [
	[5, 10,  5,  5,  0,  0,  5, 10,  0, 10, 10,  5,  5,  0,  5,  0, 10,  5,  5,  0,  5, 10, 10, 10,  5], # Day
	[5,  0,  5,  5, 10, 10,  5,  0, 10,  0,  0,  5,  5, 10,  5, 10,  0,  5,  5, 10,  5,  0,  0,  0,  5], # Night
]

day_penalties = [
	[10, 10,  5,  5,  0,  5, 10, 10, 10,  5,  5,  5,  5,  0,  5,  5,  5,  5,  0,  5,  0,  5,  5,  0,  5], # Mon
    [ 0,  5, 10,  0,  5,  5,  5,  5,  5,  0,  0,  5,  5, 10,  5, 10, 10,  0, 10,  5,  5,  0,  5,  5,  0], # Tue
    [ 5,  5,  5,  5,  5,  5,  0,  0, 10, 10,  0,  5, 10,  5,  0,  5,  5, 10,  5, 10, 10,  5, 10,  5,  5], # Wed
    [ 5,  0,  5,  0, 10, 10,  5,  5,  5,  0, 10, 10,  0, 10, 10, 10, 10,  5,  5,  0,  5, 10,  0,  0, 10], # Thur
    [ 0,  5,  0,  5,  5, 10,  5, 10,  0,  5,  5, 10,  0,  0,  5,  0,  0,  5,  0,  5,  5,  0,  5, 10, 10], # Fri
    [10, 10, 10, 10, 10,  0, 10,  0,  0, 10, 10,  0, 10,  5,  0,  5,  0,  0, 10,  0,  0,  5, 10, 10,  5], # Sat
    [ 5,  0,  0, 10,  0,  0,  0,  5,  5,  5,  5,  0,  5,  5, 10,  0,  5, 10,  5, 10, 10, 10,  0,  5,  0], # Sun
]

coworker_penalties = [
	[ 5,  5, 10,  5,  5,  5,  5,  0, 10,  0,  0,  5, 10,  5,  0,  5,  5,  5,  5, 10,  0, 10,  5,  5,  5],
    [10,  0,  5,  0,  0,  5, 10,  5,  5,  5,  5,  5, 10,  5,  5,  5,  5, 10,  5,  5,  5,  0,  5, 10,  0],
    [ 5, 10,  5, 10,  5,  5,  0,  5,  5,  0,  5,  5,  5,  5,  0,  5,  5,  0,  5,  5,  5, 10, 10,  0, 10],
	[ 5,  5,  0, 10,  5,  5,  5,  5,  5,  5,  0, 10,  5,  5,  0,  5, 10, 10,  5,  0,  5,  0,  5, 10,  5],
    [ 5,  5,  5, 10,  0, 10,  5,  5, 10,  5,  5,  0,  0,  5,  5,  0,  5,  5,  5,  5, 10,  5, 10,  5,  0],
    [ 5,  5, 10, 10,  5,  0,  5, 10,  5,  5,  5,  5,  0,  0,  0, 10,  5,  5,  5,  5,  5,  5,  0,  5, 10],
    [ 0,  0, 10,  5,  0,  0, 10,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 10, 10,  5,  5,  5, 10,  0,  5],
    [ 0, 10,  5, 10,  5,  5,  5,  5,  5,  5,  5,  5,  5, 10,  5,  0,  5,  5,  0,  5,  0,  5, 10,  0, 10],
    [10,  5,  5, 10,  0,  0,  5,  5, 10,  5,  5,  5,  5,  0,  0,  5, 10,  5,  5,  5,  5,  0,  5, 10,  5],
    [10, 10,  5,  5,  5, 10,  5,  0,  0,  0,  5, 10,  5,  5,  5,  5, 10,  0,  0,  5,  5,  5,  5,  5,  5],
    [ 0,  5,  5, 10, 10,  5,  5,  5,  5, 10,  5,  5, 10,  5,  5,  0,  0,  0, 10,  5,  0,  5,  5,  5,  5],
    [ 5,  0,  0,  5, 10,  5,  0, 10,  5,  5, 10,  5, 10, 10,  5,  5,  0,  5,  5,  5,  5,  0,  5,  5,  5],
    [ 5,  0,  5, 10, 10,  0, 10,  5,  0,  5, 10,  5,  0, 10,  5,  5,  0,  5,  5,  5,  5,  5,  5,  5,  5],
    [ 0,  5,  5, 10,  0,  5,  5,  5,  5,  5,  5,  0,  0, 10, 10,  5,  5,  5, 10,  0,  5,  5,  5, 10,  5],
    [ 5,  5, 10, 10,  5, 10,  5,  0,  5,  5,  5,  0,  5,  0,  0, 10,  5,  5,  5,  5,  5,  0,  5,  5, 10],
    [10,  5, 10,  5,  5,  5,  5,  0,  0,  5,  5,  5,  0,  5,  5,  5, 10,  5,  0,  5, 10,  5,  5,  0, 10],
    [10,  0,  5,  0,  5, 10,  5,  5,  5, 10,  5,  0,  5,  5,  5,  5, 10,  5,  5, 10,  5,  5,  0,  0,  5],
    [ 0,  5,  5,  5, 10,  5,  5,  0,  5, 10,  0, 10,  5,  5,  5,  5,  5,  0,  5,  0, 10, 10,  5,  5,  5],
    [ 5,  5, 10,  0,  5,  5, 10, 10,  0,  5,  0,  5,  0, 10, 10,  5,  5,  5,  5,  5,  5,  5,  5,  0,  5],
    [ 5,  0,  5,  5, 10, 10,  0,  5, 10,  5,  0,  5,  5,  5,  5,  0,  5,  5,  5,  0,  5,  5, 10, 10,  5],
    [ 5,  5, 10,  5,  0, 10,  5,  5,  5,  5,  5,  5,  5,  5,  0,  5,  5,  5, 10, 10,  0,  5,  0,  0, 10],
    [ 5,  5,  0, 10, 10,  5,  5,  5,  5,  0,  5,  5,  5,  5,  0,  5,  5,  0,  5,  0, 10,  5, 10, 10,  5],
    [ 5,  5,  5,  5,  5,  5,  5,  5,  0,  5, 10,  0, 10,  5,  5,  0,  5, 10,  5, 10,  5,  5, 10,  0,  0],
    [ 0,  5,  0, 10,  5,  0,  5, 10,  5,  5, 10,  5, 10,  5,  5,  5, 10,  5,  5,  5,  5,  5,  5,  0,  0],
    [ 0,  0,  0,  0,  5,  5,  5,  5,  5,  5, 10, 10,  5,  5,  5,  5,  5,  5,  5, 10, 10,  5, 10,  0,  5],
]

# day_penalties = numpy.zeros((7, num_employees))
# for d in range(7):
# 	day_penalties

def negated_bounded_span(works, start, length):
	"""Filters an isolated sub-sequence of variables assined to True.
  Extract the span of Boolean variables [start, start + length), negate them,
  and if there is variables to the left/right of this span, surround the span by
  them in non negated form.
  Args:
	works: a list of variables to extract the span from.
	start: the start to the span.
	length: the length of the span.
  Returns:
	a list of variables which conjunction will be false if the sub-list is
	assigned to True, and correctly bounded by variables assigned to False,
	or by the start or end of works.
  """
	sequence = []
	# Left border (start of works, or works[start - 1])
	if start > 0:
		sequence.append(works[start - 1])
	for i in range(length):
		sequence.append(works[start + i].Not())
	# Right border (end of works or works[start + length])
	if start + length < len(works):
		sequence.append(works[start + length])
	return sequence


def add_soft_sequence_constraint(model, works, hard_min, hard_max, prefix):
	"""Sequence constraint on true variables with soft and hard bounds.
  This constraint look at every maximal contiguous sequence of variables
  assigned to true. If forbids sequence of length < hard_min or > hard_max.
  Then it creates penalty terms if the length is < soft_min or > soft_max.
  Args:
	model: the sequence constraint is built on this model.
	works: a list of Boolean variables.
	hard_min: any sequence of true variables must have a length of at least
	  hard_min.
	soft_min: any sequence should have a length of at least soft_min, or a
	  linear penalty on the delta will be added to the objective.
	min_cost: the coefficient of the linear penalty if the length is less than
	  soft_min.
	soft_max: any sequence should have a length of at most soft_max, or a linear
	  penalty on the delta will be added to the objective.
	hard_max: any sequence of true variables must have a length of at most
	  hard_max.
	max_cost: the coefficient of the linear penalty if the length is more than
	  soft_max.
	prefix: a base name for penalty literals.
  Returns:
	a tuple (variables_list, coefficient_list) containing the different
	penalties created by the sequence constraint.
  """
	cost_literals = []
	cost_coefficients = []

	# Forbid sequences that are too short.
	for length in range(1, hard_min):
		for start in range(len(works) - length + 1):
			model.AddBoolOr(negated_bounded_span(works, start, length))

	# Just forbid any sequence of true variables with length hard_max + 1
	for start in range(len(works) - hard_max):
		model.AddBoolOr(
			[works[i].Not() for i in range(start, start + hard_max + 1)])
	return cost_literals, cost_coefficients


def add_soft_sum_constraint(model, works, hard_min, hard_max, prefix):
	"""Sum constraint with soft and hard bounds.
  This constraint counts the variables assigned to true from works.
  If forbids sum < hard_min or > hard_max.
  Then it creates penalty terms if the sum is < soft_min or > soft_max.
  Args:
	model: the sequence constraint is built on this model.
	works: a list of Boolean variables.
	hard_min: any sequence of true variables must have a sum of at least
	  hard_min.
	soft_min: any sequence should have a sum of at least soft_min, or a linear
	  penalty on the delta will be added to the objective.
	min_cost: the coefficient of the linear penalty if the sum is less than
	  soft_min.
	soft_max: any sequence should have a sum of at most soft_max, or a linear
	  penalty on the delta will be added to the objective.
	hard_max: any sequence of true variables must have a sum of at most
	  hard_max.
	max_cost: the coefficient of the linear penalty if the sum is more than
	  soft_max.
	prefix: a base name for penalty variables.
  Returns:
	a tuple (variables_list, coefficient_list) containing the different
	penalties created by the sequence constraint.
  """
	cost_variables = []
	cost_coefficients = []
	sum_var = model.NewIntVar(hard_min, hard_max, '')
	# This adds the hard constraints on the sum.
	model.Add(sum_var == sum(works))

	return cost_variables, cost_coefficients


def solve_shift_scheduling(params, output_proto):
	"""Solves the shift scheduling problem."""

	# The required number of shifts worked per week (min, max)
	max_shifts_per_week_constraint = (0, 4)
	
	# The required number of day shifts in a 2 week schedule (min, max)
	day_shifts_per_two_weeks = (1, num_days * num_shifts)
	
	# Number of nurses that MUST be assigned to each shift (min, max)
	required_nurses_per_shift = (7, 7)

	model = cp_model.CpModel()

	work = {}
	for e in range(num_employees):
		for d in range(num_days):
			for s in range(num_shifts):
				work[e, d, s] = model.NewBoolVar('work%i_%i_%i' % (e, d, s))


	# Handle the required nurses per shift constraint
	hard_min, hard_max = required_nurses_per_shift
	for s in range(num_shifts):
		for w in range(num_weeks):
			for d in range(7):
				works = [work[e, w * 7 + d, s] for e in range(num_employees)]
				# Ignore Off shift.
				worked = model.NewIntVar(hard_min, hard_max, '')
				model.Add(worked == sum(works))
				
	# Handle the min day shifts per 2 weeks constraint
	# hard_min, hard_max = day_shifts_per_two_weeks
	# for e in range(num_employees):
		# works = [work[e, d, day_shift] for d in range(num_days)]
		# variables, coeffs = add_soft_sum_constraint(
				# model, works, hard_min, hard_max,
				# 'weekly_sum_constraint(employee %i, day shift)' %
				# (e))

	# Handle the max shifts per week constraint
	hard_min, hard_max = max_shifts_per_week_constraint
	for e in range(num_employees):
		for w in range(num_weeks):
			works = [work[e, d + w * 7, s] for d in range(7) for s in range(num_shifts)]
			variables, coeffs = add_soft_sum_constraint(
				model, works, hard_min, hard_max,
				'weekly_sum_constraint(employee %i, shift %i, week %i)' %
				(e, s, w))
				

	if output_proto:
		print('Writing proto to %s' % output_proto)
		with open(output_proto, 'w') as text_file:
			text_file.write(str(model))

	# Solve the model.
	solver = cp_model.CpSolver()
	if params:
		text_format.Parse(params, solver.parameters)
	solution_printer = VarArraySolutionPrinterWithLimit(solver, work, solutions_to_find)
	status = solver.SearchForAllSolutions(model, solution_printer)

	print()
	print('Statistics')
	print('	 - status		   : %s' % solver.StatusName(status))
	print('	 - conflicts	   : %i' % solver.NumConflicts())
	print('	 - branches		   : %i' % solver.NumBranches())
	print('	 - wall time	   : %f s' % solver.WallTime())
	return solution_printer.get_solutions()

class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
	"""Print intermediate solutions."""

	def __init__(self, solver, work, limit):
		cp_model.CpSolverSolutionCallback.__init__(self)
		self.__solver = solver
		self.__work = work
		self.__solution_count = 0
		self.__solution_limit = limit
		self.__solutions = []

	def on_solution_callback(self):
		self.__solution_count += 1
			
		
		self.__solutions.append([self.Value(self.__work[e, d, s]) for e in range(num_employees) for d in range(num_days) for s in range(num_shifts)])
		
		# PRINTS THE 30 INITIAL SCHEDULES
		header = '		   '
		for w in range(num_weeks):
			header += 'M	 T	 W	 T	 F	 S	 S	 '
		print(header)
		for e in range(num_employees):
			schedule = ''
			for d in range(num_days):
				schedule_to_add = ''
				for s in range(num_shifts):
					if self.Value(self.__work[e, d, s]):
						schedule_to_add += shifts[s]
				schedule += schedule_to_add.ljust(4)
			print('worker %2i: %s' % (e, schedule))
		print()
	
		if self.__solution_count >= self.__solution_limit:
			print('Stop search after %i solutions' % self.__solution_limit)
			self.StopSearch()

	def solution_count(self):
		return self.__solution_count
		
	def get_solutions(self):
		return self.__solutions
	
def main(_):
	solutions = solve_shift_scheduling(FLAGS.params, FLAGS.output_proto)

	print("\nStarting GWO\n")
	GWO.GWO(solutions, Fitness, 0, 1, interactions, PrintSchedule, 
	gwoScore_x_iterations, gwoScore_x_time, gwoCPU_x_iterations, gwoRAM_x_iterations)

	print("Starting MFO\n")
	MFO.MFO(solutions, Fitness, 0, 1, interactions+1, PrintSchedule, 
	mfoScore_x_iterations, mfoScore_x_time, mfoCPU_x_iterations, mfoRAM_x_iterations)

	print("Starting MVO\n")
	MVO.MVO(solutions, Fitness, 0, 1, interactions, PrintSchedule, 
	mvoScore_x_iterations, mvoScore_x_time, mvoCPU_x_iterations, mvoRAM_x_iterations)

	#PLOT THE RESULTS IN GRAPHS

	#Figure 1 - Fitness Score Vs Iterations
	plt.figure(1)
	# naming the x axis
	plt.xlabel('x - Iterations')
	# naming the y axis
	plt.ylabel('y - Fitness Score')
	# giving a title to my graph
	plt.title('Fitness Score Vs Iterations')

	# plotting the Moth Flame vs Iterations
	plt.plot(interactionsArray, mfoScore_x_iterations, label = "Moth Flame Optimizer")
	
	# plotting the Grey Wolf vs Iterations
	plt.plot(interactionsArray, gwoScore_x_iterations, label = "Grey Wolf Optimizer")

	# plotting the Grey Wolf vs Iterations
	plt.plot(interactionsArray, mvoScore_x_iterations, label = "Multiverse Optimizer")
	
	# show a legend on the plot
	plt.legend()
	
	#Figure 2 - Fitness Score Vs Time
	plt.figure(2)
	# naming the x axis
	plt.xlabel('x - Time (s)')
	# naming the y axis
	plt.ylabel('y - Fitness Score')
	# giving a title to my graph
	plt.title('Fitness Score Vs Time')
	# plotting the Moth Flame vs Iterations
	plt.plot(interactionsArray, mfoScore_x_iterations, label = "Moth Flame Optimizer")
	
	# plotting the Grey Wolf vs Iterations
	plt.plot(interactionsArray, gwoScore_x_iterations, label = "Grey Wolf Optimizer")

	# plotting the Multiverse vs Iterations
	plt.plot(interactionsArray, mvoScore_x_iterations, label = "Multiverse Optimizer")

	
	# show a legend on the plot
	plt.legend()
	
	#Figure 3 - RAM Vs Iterations
	plt.figure(3)
	# naming the x axis
	plt.xlabel('x - Iterations')
	# naming the y axis
	plt.ylabel('y - RAM')
	# giving a title to my graph
	plt.title('RAM Vs Iterations')
	# plotting the Moth Flame vs Iterations
	plt.plot(interactionsArray, mfoRAM_x_iterations, label = "Moth Flame Optimizer")
	
	# plotting the Grey Wolf vs Iterations
	plt.plot(interactionsArray, gwoRAM_x_iterations, label = "Grey Wolf Optimizer")

	# plotting the Multiverse vs Iterations
	plt.plot(interactionsArray, mvoRAM_x_iterations, label = "Multiverse Optimizer")

	
	# show a legend on the plot
	plt.legend()

	#Figure 4 - CPU Vs Iterations
	plt.figure(4)
	# naming the x axis
	plt.xlabel('x - Iterations')
	# naming the y axis
	plt.ylabel('y - CPU')
	# giving a title to my graph
	plt.title('CPU Vs Iterations')
	# plotting the Moth Flame vs Iterations
	plt.plot(interactionsArray, mfoCPU_x_iterations, label = "Moth Flame Optimizer")
	
	# plotting the Grey Wolf vs Iterations
	plt.plot(interactionsArray, gwoCPU_x_iterations, label = "Grey Wolf Optimizer")

	# plotting the Multiverse vs Iterations
	plt.plot(interactionsArray, mvoCPU_x_iterations, label = "Multiverse Optimizer")

	
	# show a legend on the plot
	plt.legend()

	# function to show the plot
	plt.show()

def PrintSchedule(schedule):
	header = '           '
	for w in range(1):
		header += 'M   T   W   T   F   S   S   M   T   W   T   F   S   S'
	print(header)
	for e in range(num_employees):
		schedule_text = ''
		for d in range(num_days):
			schedule_to_add = ''
			i = e * (num_days*num_shifts) + d * num_shifts
			if schedule[i] == 1:
				schedule_to_add += 'D'
			if schedule[i+1] == 1:
				schedule_to_add += 'N'
			schedule_text += schedule_to_add.ljust(4)
		print('worker %2i: %s' % (e, schedule_text))
	print()
	
def CheckValidity(schedule):
	# Check for correct number of nurses assigned to shifts
	for d in range(num_days):
		for s in range(num_shifts):
			nurses_on_shift = [schedule[i] for i in range(len(schedule)) if (i % (num_days * num_shifts)) // num_shifts == d and i % num_shifts == s].count(1)
			if nurses_on_shift != 7:
				return False	

	for nurse in range(num_employees):
		day_shift_req_met = False
		for i in range(nurse * num_days * num_shifts, (nurse+1) * num_days * num_shifts, 2):
			if schedule[i] == 1:
				day_shift_req_met = True
				break
		if not day_shift_req_met:
			return False
		
		for week in range(num_weeks):
			shift_count = 0
			for i in range(nurse * week * 7 * num_shifts, nurse * week * 7 * num_shifts + 7 * num_shifts):
				if schedule[i] == 1:
					shift_count += 1
			if shift_count > 4:
				return False

	return True

	
def NurseFitness(nurse, schedule):
	# Arbitrary implementation of a fitness function.
	# A nurse wants to work on 1 specific day. If assigned to work any other day
	# then add 10 points (overall goal is to minimize this number)

	nurse_index_range = range(nurse*num_days*num_shifts, (nurse+1)*num_days*num_shifts)

	sum = 0
	for i in nurse_index_range:

		# 0-14 2 WEEKS
		day = (i % (num_days * num_shifts)) // num_shifts

		# DAY/NIGHT
		shift = i % num_shifts

		# VIOLATION 1 BACK TO BACK SHIFTS
		if i + 1 < nurse_index_range.stop:
			if schedule[i] == 1 and schedule[i+1] == 1:
				sum += 10

		# VIOLATION 2 DAY OF PREFERENCE
		if schedule[i] == 1:
			sum += day_penalties[day % 7][nurse]

		# VIOLATION 3 PREFERENCE FOR NIGHT OF DAY SHIFTS
		if schedule[i] == 1:
			sum += shift_penalties[shift][nurse]

		# VIOLATION 4 PREFERENCE TO WORK WITH ANOTHER NURSE
		if schedule[i] == 1:
			for otherNurse in range(num_employees):
				if schedule[otherNurse*num_days*num_shifts + day*num_shifts + shift] == 1:
					sum += coworker_penalties[otherNurse][nurse]
				else:
					sum += 10 - coworker_penalties[otherNurse][nurse]
	return sum
	
def Fitness(schedule):
	sum = 0

	if CheckValidity(schedule):
		for nurse in range(num_employees):
			sum += NurseFitness(nurse, schedule)

	else: 
		sum = 99999
		return sum

	return sum


if __name__ == '__main__':
	app.run(main)