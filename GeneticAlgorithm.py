import numpy as np

class GeneticAlgorithm:
	def __init__(self, n_kromosom_, n_gen_, crossover_rate_, mutation_rate_, constraints_, obj_func_const_):
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

		self.max_fitness_before = 0.0
		self.chromosome = np.zeros(shape=(self.n_kromosom, self.n_gen))
		self.val_obj_func = np.zeros(shape=(self.n_kromosom, 1))


		self.initialize_chromosome()

	def initialize_chromosome(self):
		self.chromosome = np.expand_dims(self.chromosome, axis=1)
		print(self.chromosome.shape)
		for n_c in range(len(self.constraints)):
			self.chromosome[:,0,n_c] = np.random.randint(self.constraints[n_c][0], self.constraints[n_c][1], size=(self.n_kromosom))
		self.chromosome = np.concatenate(self.chromosome, axis=0)

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

		if max_fitness > 0.99:
			return_val[0] = True
			return return_val
		else:
			roullete_wheel_resampling(self)
			return return_val

	def one_point_crossover(self):
		n_induk = int(self.n_kromosom * self.cr)
		index = np.random.randint(0,self.n_kromosom, (n_induk))
		self.induks = self.chromosome[index]
		# self.induks = self.chromosome[self.n_gen - n_induk -1: self.n_gen]

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
		# self.chromosome[self.n_gen - n_induk -1: self.n_gen] = self.chrom_after_crossover


	def mutation(self):
		self.total_gen = self.n_kromosom * self.n_gen
		n_mutation = int(self.total_gen * self.mr)

		for m in range(n_mutation):
			ind_mutation = np.random.randint(n_mutation)
			chrom_mutated = int(ind_mutation/self.n_gen)
			gen_mutated = ind_mutation % self.n_gen
			self.chromosome[chrom_mutated][gen_mutated] = np.random.randint(self.constraints[gen_mutated][1])

	def train(self, max_iter):
		generation = 0
		while True:
			[finish, max_fit, solution] = self.selection()
			if max_fit > self.max_fitness_before:
				self.max_fitness_before = max_fit
				self.solution = solution

			generation += 1
			print("Generation-", generation, "\tfitness: ", self.max_fitness_before, "\t solution: ", self.solution)
			if finish or (generation > max_iter):
				break
			self.one_point_crossover()
			self.mutation()



n_kromosom = 100
obj_func_const = np.array([1,2,3,4,-30])
n_gen = obj_func_const.shape[0]-1
constraints = [[0,30], [0,10], [0,10], [0,10]]
crossover_rate = 0.7
mutation_rate = 0.5

GA = GeneticAlgorithm(n_kromosom, n_gen, crossover_rate, mutation_rate, constraints, obj_func_const)
GA.train(500)
