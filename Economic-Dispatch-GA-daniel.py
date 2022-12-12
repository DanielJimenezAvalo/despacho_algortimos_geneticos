import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
class GeneticAlgorithm:
	def __init__(self, 
              n_kromosom_, 
              n_gen_, 
              crossover_rate_, 
              mutation_rate_, 
              desired_fitness_, 
              constraints_, 
              obj_func_const_):
        
		self.n_kromosom = n_kromosom_
		self.n_gen = n_gen_
		self.constraints = constraints_
		self.const_obj_func = obj_func_const_
		self.val_obj_func = None
		self.total_fitness = 0.0
		self.prob_fitness = np.zeros(shape=(self.n_kromosom,1))
		self.cdf = np.zeros(shape=(self.n_kromosom,1))
		self.cr = crossover_rate_
		self.mr = mutation_rate_
		self.desired_fitness = desired_fitness_

		self.max_fitness_before = 0.0
		self.chromosome = np.zeros(shape=(self.n_kromosom, self.n_gen))
		self.val_obj_func = np.zeros(shape=(self.n_kromosom, 1))


		self.initialize_chromosome()

	def initialize_chromosome(self):
		self.chromosome = np.expand_dims(self.chromosome, axis=1)
		for n_c in range(len(self.constraints)):
			self.chromosome[:,0,n_c] = np.random.uniform(self.constraints[n_c][0], self.constraints[n_c][1], size=(self.n_kromosom))
		self.chromosome = np.concatenate(self.chromosome, axis=0)
		# print(self.chromosome, "\n\n")

	def selection(self):
		def obj_func(self):
			self.val_obj_func = np.abs(np.sum(self.const_obj_func[0:-1] * self.chromosome, axis= -1) + self.const_obj_func[-1])

		def fitness(self):
			self.val_fitness = 1/(self.val_obj_func + 1.0)
			self.total_fitness = np.sum(self.val_fitness)

		def prob_based_fitness(self):
			self.prob_fitness = self.val_fitness / self.total_fitness

		def cdf(self):
			for l in range(self.n_kromosom):
				if l == 0:
					self.cdf[l] = self.prob_fitness[l]
				else:
					self.cdf[l] = self.cdf[l-1] + self.prob_fitness[l]
		
		random_selector = np.random.uniform(low=0.0, high=1.0, size=(self.n_kromosom))
		selected_chrom = self.chromosome

		def roullete_wheel_resampling(self):
			prob_based_fitness(self)
			cdf(self)

			for r in range(random_selector.shape[0]):
				for c in range(self.cdf.shape[0]):
					if c == 0:
						if random_selector[r] < self.cdf[c]:
							selected_chrom[r] = self.chromosome[c]
					else:
						if (self.cdf[c-1] < random_selector[r]) & (random_selector[r] < self.cdf[c]):
							selected_chrom[r] = self.chromosome[c]
			self.chromosome = selected_chrom

		obj_func(self)
		fitness(self)

		max_fitness = np.amax(self.val_fitness)
		ind_max_fit = np.where(self.val_fitness == max_fitness)
		ind_max_fit = list(list(set(zip(*ind_max_fit)))[0])[0]

		return_val = [False, max_fitness, self.chromosome[ind_max_fit]]

		if max_fitness > self.desired_fitness:
			return_val[0] = True
			return return_val
		else:
			roullete_wheel_resampling(self)
			return return_val

	def one_point_crossover(self):
		n_induk = int(self.n_kromosom * self.cr)
		index = np.random.randint(0,self.n_kromosom, (n_induk))
		self.induks = self.chromosome[index]

		couple_index = list(np.arange(self.induks.shape[0]))
		couple_index.append(0)
		temp = couple_index
		couple_index = [temp[c:c+2] for c in range(len(temp)-1)]
		couples = [self.induks[couple_index[v]] for v in range(len(couple_index))]

		self.chrom_after_crossover = self.induks
		for q in range(len(couples)):
			index_ = np.random.randint(self.n_gen)
			self.chrom_after_crossover[q][0:index_]= couples[q][0][0:index_]
			self.chrom_after_crossover[q][index_:self.n_gen]= couples[q][1][index_:self.n_gen]
		self.chromosome[index] = self.chrom_after_crossover


	def mutation(self):
		self.total_gen = self.n_kromosom * self.n_gen
		n_mutation = int(self.total_gen * self.mr)

		for m in range(n_mutation):
			ind_mutation = np.random.randint(n_mutation)
			chrom_mutated = int(ind_mutation/self.n_gen)
			gen_mutated = ind_mutation % self.n_gen
			self.chromosome[chrom_mutated][gen_mutated] = np.random.uniform(self.constraints[gen_mutated][1])


Pi = list()
Pi.append(np.array([32,50,87,94]))
Pi.append(np.array([21,34,100,122]))
Pi.append(np.array([15,26,54,108]))
Pi.append(np.array([22,45,55,127]))
Pi.append(np.array([19,47,58,97]))
Pi.append(np.array([19,33,72,147]))
Pi.append(np.array([20,42,74,104]))
Pi.append(np.array([24,36,60,100]))
Pi.append(np.array([19,48,80,133]))
Pi.append(np.array([23,49,68,108]))
Pi.append(np.array([16,50,78,146]))
Pi.append(np.array([25,55,81,138]))

IHR = list()
IHR.append(np.array([62,95,136,175]))
IHR.append(np.array([61,88,140,178]))
IHR.append(np.array([69,111,141,152]))
IHR.append(np.array([60,118,143,191]))
IHR.append(np.array([41,119,142,172]))
IHR.append(np.array([47,81,129,167]))
IHR.append(np.array([46,102,134,186]))
IHR.append(np.array([49,115,141,169]))
IHR.append(np.array([52,84,138,178]))
IHR.append(np.array([67,94,143,182]))
IHR.append(np.array([46,111,141,197]))
IHR.append(np.array([51,80,149,185]))

fuel_cost = list()
fuel_cost.append(0.084)
fuel_cost.append(0.268)
fuel_cost.append(0.265)
fuel_cost.append(0.756)
fuel_cost.append(0.12)
fuel_cost.append(0.043)
fuel_cost.append(0.124)
fuel_cost.append(0.436)
fuel_cost.append(0.18)
fuel_cost.append(0.775)
fuel_cost.append(0.395)
fuel_cost.append(0.059)

cost = list()
for x in range(len(IHR)):
	cost.append(IHR[x]*fuel_cost[x])#/1000

def least_square(P_, b_):
	A_raw = np.array([P_**2, P_, P_**0])
	A = np.transpose(A_raw)
	At = A_raw
	AtA = np.dot(At,A)
	Atb = np.dot(At,b_)
	invAtA = np.linalg.inv(AtA)
	abc = np.dot(invAtA,Atb)
	return abc

power_min = list()
power_max = list()
cost_min = list()
cost_max = list()
for l in range(len(Pi)):
	power_min.append(np.amin(Pi[l]))
	power_max.append(np.amax(Pi[l]))
	cost_min.append(np.amin(cost[l]))
	cost_max.append(np.amax(cost[l]))

abc = list()
for y in range(len(cost)):
	abc.append(least_square(Pi[y], cost[y]))

arange = list()
est_cost_arange = list()
for m in range(len(power_max)):
	delta_ = (power_max[m] - power_min[m])/100.
	arange.append(np.arange(power_min[m], power_max[m], delta_))
	temp_ = abc[m][0]*arange[-1]**2 + abc[m][1]*arange[-1] + abc[m][2]*arange[-1]**0
	est_cost_arange.append(temp_)

fig, ax = plt.subplots()
ax.plot(arange[0], est_cost_arange[0], '-g', label='G1')
ax.plot(arange[1], est_cost_arange[1], '-b', label='G2')
ax.plot(arange[2], est_cost_arange[2], '-k', label='G3')
ax.plot(arange[3], est_cost_arange[3], '-r', label='G4')
ax.plot(arange[4], est_cost_arange[4], '-c', label='G5')
ax.plot(arange[5], est_cost_arange[5], '-m', label='G6')
ax.plot(arange[6], est_cost_arange[6], '-y', label='G7')
ax.plot(arange[7], est_cost_arange[7], '-w', label='G8')
ax.plot(arange[8], est_cost_arange[8], '-g', label='G9')
ax.plot(arange[9], est_cost_arange[9], '-b', label='G10')
ax.plot(arange[10], est_cost_arange[10], '-k', label='G11')
ax.plot(arange[11], est_cost_arange[11], '-r', label='G12')
legend = ax.legend(loc='upper right', shadow=True)

n_kromosom = 50
obj_func_const = np.array([1,1,1,1,1,1,1,1,1,1,1,1,-200])
n_gen = obj_func_const.shape[0]-1
constraints = [[power_min[0],power_max[0]], 
               [power_min[1],power_max[1]], 
               [power_min[2],power_max[2]],
               [power_min[3],power_max[3]],
               [power_min[4],power_max[4]],
               [power_min[5],power_max[5]],
               [power_min[6],power_max[6]],
               [power_min[7],power_max[7]],
               [power_min[8],power_max[8]],
               [power_min[9],power_max[9]],
               [power_min[10],power_max[10]],
               [power_min[11],power_max[11]]]
               
crossover_rate = 0.9
mutation_rate = 0.7
desired_fitness = 0.95

GA = GeneticAlgorithm(n_kromosom, n_gen, crossover_rate, mutation_rate, desired_fitness, constraints, obj_func_const)

part_ = 1
generation = 0
max_iter = 200
size_to_compare = 50
list_solution = list()
list_cost = list()
list_arr_cost = list()
finish = False
while True:
	[finish, max_fit, solution] = GA.selection()
	generation += 1
	print("Part-", part_, "\tGeneration-", generation, "\tfitness: ", max_fit, "\t solution: ", solution)
	if finish or (generation > max_iter):
		if len(list_solution) < size_to_compare:
			cost_ = list()
			for k in range(len(cost)):
				cost_.append(abc[k][0]*solution[k]**2 + abc[k][1]*solution[k] + abc[k][2]*solution[k]**0)
			total_cost = np.sum(cost_)
			if (total_cost >= np.sum(cost_min)) & (total_cost <= np.sum(cost_max)):
				list_cost.append(total_cost)
				list_arr_cost.append(cost_)
				list_solution.append(solution)
			part_+=1
			generation = 0
			GA = GeneticAlgorithm(n_kromosom, n_gen, crossover_rate, mutation_rate, desired_fitness, constraints, obj_func_const)
			[finish, max_fit, solution] = GA.selection()
		else:
			break
	GA.one_point_crossover()
	GA.mutation()

min_cost = np.amin(list_cost)
ind_min_cost = np.where(list_cost == min_cost)
ind_min_cost = list(list(set(zip(*ind_min_cost)))[0])[0]
fix_solution = list_solution[ind_min_cost]
fix_arr_cost = list_arr_cost[ind_min_cost]

print("costo_min_total: ", np.sum(cost_min), "\tcosto_max_total: ", np.sum(cost_max), "\n")
print("el menor costo de operacion es: ", fix_solution, "\tcon un costo de: ", fix_arr_cost, "\tcosto_total: ", min_cost)

delta_time = time.time() - start_time
print("tiempo de ejecucion: ", delta_time)

ax.plot(fix_solution[0], fix_arr_cost[0], '^g', fix_solution[1], fix_arr_cost[1], '^b', fix_solution[2], fix_arr_cost[2], '^k')
plt.xlabel("Potencia-i (MW)")
plt.ylabel("Costo ($/kWh)")
plt.show()
