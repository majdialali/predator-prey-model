import random as random1  # importing as random created name conflict
import numpy as np
import pycxsimulator
from pylab import *
import math
from matplotlib import colors
import matplotlib.pyplot as plt
import nn_toolbox  # Another script


class RabbitClass:
    """ Rabbit class to represent each rabbit """
    def __init__(self,
                 grid_size,
                 speed=int(round(8 + np.random.randn() * 2, 0)),
                 mass=int(round(4 + np.random.randn() * 2, 0)),
                 list_of_experiences= [],
                 list_of_experienced_consequences= [],
                 health=500):
        """
            Description
                Initialise Rabbit class instance

            Arguments:
                grid_size - integer, size of grid in simulation
                speed - integer, speed of rabbits
                mass - integer, mass of rabbits
                list_of_experiences - list
                list_of_experienced_consequences, list
                health - float, health of rabbit at birth

            Returns:
                none
        """
        self.speed = int(round(speed, 0))
        if self.speed < 1:  # Lower limit for speed is 1
            self.speed = 1
        self.health = health
        self.mass = int(round(mass, 0))
        if self.mass < 1:  # Lower limit for mass is 1
            self.mass = 1
        if self.mass > 10:  # Lower limit for mass is 1
            self.mass = 10
        self.digestion_efficiency = 0.5 / (1 + np.exp(-2 * self.mass + 7)) + 0.5  # Sigmoid like function
        self.health = health
        self.layer_dims = [6, 4, 1]  # 11 inputs, 2 hidden nodes in 1st and only hidden layer, 1 output node
        self.list_of_experiences = list_of_experiences
        self.list_of_experienced_consequences = list_of_experienced_consequences
        self.weights = nn_toolbox.initialize_parameters_deep(self.layer_dims)
        if self.list_of_experiences != []:  # If born with memories, learn from them at initiation
            RabbitClass.learn(self)
        self.position = [random1.randint(0, grid_size-1), random1.randint(0, grid_size-1)]  # Random position
        self.grid_size = grid_size
        self.iq = 50  # Default IQ
        self.genotype = []  # Default genotype

    def get_speed(self):
        """ Return rabbit speed """
        return self.speed

    def get_mass(self):
        """ Return rabbit mass """
        return self.mass

    def set_speed(self, speed):
        """ Set rabbit speed """
        self.speed = speed

    def get_position(self):
        """ Return rabbit position """
        return self.position[0], self.position[1]

    def get_health(self):
        """ Return rabbit health """
        return self.health

    def get_genotype(self):
        """ Return rabbit genotype """
        return self.genotype

    def get_iq(self):
        """ Return rabbit IQ """
        return self.iq

    def update_genotype(self):
        """ Update rabbit IQ """
        self.genotype = [self.speed, self.mass, [self.list_of_experiences, self.list_of_experienced_consequences]]

    def update_iq(self, all_flower_information):
        """
            Description
                Benchmark rabbit intelligence and save it as rabbit IQ
            Arguments:
                all_flower_information - dict, information on flowers
            Returns:
                accuracy - float, 0..100, rabbit intelligence
        """
        # Construct training examples
        for flower in all_flower_information.values():  # Loop for flower
            for flower_size in range(1, 5+1):  # Loop for flower size
                flower_with_size = flower.copy()
                flower_with_size['nutrition value'] = flower['nutrition value'][flower_size - 1]
                flower_with_size['flower size'] = flower_size

                x = RabbitClass.encode_input(self, flower_with_size)
                if flower_with_size['nutrition value'] > 0:
                    y = 1
                else:
                    y = 0
                # Add experience
                self.list_of_experiences.append(list(x[0]))
                self.list_of_experienced_consequences.append(y)
        # Format experiences and experienced consequence to numpy array
        X = np.array(self.list_of_experiences)
        Y = np.array(self.list_of_experienced_consequences)
        X = X.T
        Y = Y.reshape((X.shape[1], 1))
        Y = np.squeeze(Y.T)
        Y_p = nn_toolbox.predict(X, self.weights)

        # Calculate accuracy of prediction
        m = len(Y)
        P = np.sum(Y)
        N = m - P
        Tp = np.dot(Y_p.T, Y)
        Fp = np.sum(Y_p) - Tp
        Tn = N - Fp
        accuracy = np.round(100 * (Tp + Tn) / m, 2)
        self.iq = accuracy

        # To avoid CPU overflow, restrict number of experiences. Only remember the 1000 last experiences and
        # consequences, i.e., replace newest with oldest
        if len(self.list_of_experiences) > 1000:
            self.list_of_experiences = self.list_of_experiences[-1000:]
            self.list_of_experienced_consequences = self.list_of_experienced_consequences[-1000:]

        return accuracy

    def randomly_move(self):
        """ Randomly move rabbit 1 step, in either x or y direction, with periodic boundary condition """
        # 50/50 chance for x or y direction movement
        if random1.randint(0, 1) == 0:  # Random movement x direction
            self.position[0] += [-1, 1][random1.randint(0, 1)]
        else:  # Random movement y direction
            self.position[1] += [-1, 1][random1.randint(0, 1)]
        # Control for max position
        for i in range(0, 2):
            if self.position[i] >= self.grid_size:  # Jump to other side
                self.position[i] = 0
            elif self.position[i] == -1:
                self.position[i] = grid_size-1

    def update_health_with_food(self, nutrition_value):
        """
            Description
                Update rabbit health with flower nutrition based on digestion efficiency
            Arguments:
                nutrition value - float, nutrition of flower
            Returns:
                none
        """
        self.health += nutrition_value * self.digestion_efficiency

    def update_health(self):
        """ Update rabbit health based on energy consumption during simulation iteration """
        self.health -= 1 / 2 * self.mass * self.iq / 100 * self.speed ** 2

    def encode_input(self, flower_information):
        """
            Description
                Encode flower information to ANN input, i.e., numpy array
            Arguments:
                flower_information - dict, information of a single flower
            Returns:
                X - numpy array, input to ANN
        """
        X = np.zeros((6, 1))

        if flower_information['color'] == 'yellow':
            X[0, 0] = 1
        if flower_information['color'] == 'red':
            X[1, 0] = 1
        if flower_information['color'] == 'blue':
            X[2, 0] = 1
        if flower_information['color'] == 'green':
            X[3, 0] = 1
        if flower_information['color'] == 'black':
            X[4, 0] = 1

        X[5, 0] = (5 - flower_information['flower size']) / 4 - 5 / 10  # Apply feature normalisation

        X = X.T
        return X

    def decide(self, flower_at_position):
        """
            Description
                Decide to eat a flower based on its color and size
            Arguments:
                flower_at_position - dict, information of a single flower
            Returns:
                choice - bool, choice, True, do it, False, do not do it
        """
        # Random choice
        #list_of_choice = [True, False]
        #return list_of_choice[random1.randint(0, 1)]
        # Choice using ANN
        X = RabbitClass.encode_input(self, flower_at_position)
        X = X.T
        choice = nn_toolbox.predict(X, self.weights)
        choice = bool(int(choice))
        return choice

    def learn(self, flower_at_position={}, y=0):
        """
            Description
                Learn, i.e., update ANN weights based on experiences. Update experience with flower_at_position if
                not empty
            Arguments:
                flower_at_position - dict, information of a single flower
            Returns:
                none
        """
        if flower_at_position != {}:  # Update memory
            x = RabbitClass.encode_input(self, flower_at_position)
            # Add experience and consesquence
            self.list_of_experiences.append(list(x[0]))
            self.list_of_experienced_consequences.append(y)
        # Format experiences and experienced consequence to numpy array
        X = np.array(self.list_of_experiences)
        Y = np.array(self.list_of_experienced_consequences)
        X = X.T
        Y = Y.reshape((X.shape[1], 1))
        Y = Y.T
        # Learn from new experience and old ones
        self.weights = nn_toolbox.L_layer_model(X, Y, self.layer_dims, self.weights)


def observe():
    global environment_with_rabbits, plot_series
    """
    Description
        Visualises the Predator-prey environment, the population sizes and predator features

    Arguments(through global variables):
        environment_with_rabbits - numpy array, of the environment with rabbits
        plot_series - dict, population sizes and predator features

    Returns:
        none
    """

    cla()
    plt.clf()  # Reset plot environment

    plt.subplot(121)  # Environment
    cmap = colors.ListedColormap(['red', 'white', 'black'])  # Define custom colormap
    bounds = [-10, 0, 0.5, 10]  # Custom colormap bounds
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # Rabbits are shown in red, flowers in black, and vacant ground in white
    imshow(environment_with_rabbits, vmin=-1, vmax=2, cmap=cmap, norm=norm)
    axis('image')

    plt.subplot(322)  # Rabbit population
    plt.ylabel('Rabbit population')
    plt.plot(plot_series['Rabbit population'], 'r', label='Rabbit population')

    plt.subplot(324)  # Flower population
    plt.ylabel('Flower population')
    plt.plot(plot_series['Flower population'], 'k', label='Flower population')

    plt.subplot(326)  # Rabbit features
    plt.ylabel('Average rabbit feature')
    plt.plot(plot_series['Average rabbit IQ'], 'b--', label='Avg. Rabbit IQ')
    plt.plot(np.array(plot_series['Average rabbit speed']) * 10, 'r--', label='10 Avg. Rabbit Speed')
    plt.plot(np.array(plot_series['Average rabbit mass']) * 20, 'k--', label='20 Avg. Rabbit Mass')
    plt.legend()


def initialize():
    global environment, rabbits, flower_information, grid_size, plot_series
    """
     Description
         Initialise the simulation, i.e., Predator-prey environment and population of rabbits and flowers

     Arguments(through global variables):
         environment - numpy array, of the environment without rabbits
         rabbits - list, list of rabbit class instances
         flower_information - dict, with flower information
         grid_size - integer, the size of the grid 
         plot_series - dict, population sizes and predator features

     Returns(through global variables):
        environment - numpy array, of the environment without rabbits
        environment_with_rabbits - numpy array, of the environment with rabbits
        rabbits - list, list of rabbit class instances
        plot_series - dict, population sizes and predator features
     """

    # Initialize environment
    initial_flower_probability = 0.05
    # Flower information contains information on each flower. A flower is known by its key.
    # It is also represented in the environment by this key. Negative nutritional value indicates toxicness.
    environment = np.zeros((grid_size, grid_size))
    environment_with_rabbits = environment
    list_1 = list(flower_information.keys())
    # Place flowers at random positions according to initial_flower_probability
    for i in range(grid_size):
        for j in range(grid_size):
            if random1.random() < initial_flower_probability:
                # Spawn a random flower
                environment[i, j] = list_1[random1.randint(0, len(list_1)-1)]

    # Initialize population(rabbits)
    initial_population_size = 100
    rabbits = []
    for _ in range(initial_population_size):
        rabbits.append(RabbitClass(grid_size))

    # Initialize plot series
    plot_series = {'Rabbit population': [],
                   'Flower population': [],
                   'Average rabbit IQ': [],
                   'Average rabbit speed': [],
                   'Average rabbit mass': []}


def update():
    global rabbits, flower_information, environment, mutation_probability, plot_series, \
        environment_with_rabbits, rabbit_births_per_generation, crossover_probability, iteration
    """
         Description
             Updates the simulation, i.e., Predator-prey environment and population of rabbits and flowers

         Arguments(through global variables):
             rabbits - list, list of rabbit class instances
             flower_information - dict, information on flowers
             environment - numpy array, of the environment without rabbits
             mutation_probability - float, evolutionary parameter for rabbit mutation
             plot_series - dict, population sizes and predator features
             environment_with_rabbits - numpy array, of the environment with rabbits
             rabbit_births_per_generation - integer, number of rabbit births per simulation iteration
             crossover_probability - float, evolutionary parameter for rabbit crossover
             iteration - integer, simulation iteration counter
             
         Returns(through global variables):
             rabbits - list, list of rabbit class instances
             plot_series - dict, population sizes and predator features
             environment_with_rabbits - numpy array, of the environment with rabbits
             iteration - integer, simulation iteration counter
         """
    def update_environment(environment, iteration):
        """
            Description
                Updates the environment, i.e., spawn random flowers at random locations

            Arguments:
                environment - numpy array, of the environment without rabbits
                iteration - integer, simulation iteration counter

            Returns:
                environment - numpy array, of the environment without rabbits
        """
        # Count number of flowers in environment
        desired_flower_count = 300  # By default
        if iteration > 50:  # After 50 iterations, start periodic variation
            desired_flower_count = 200 + 100 * np.sin(iteration / 3)
        # Loop until minimum desired population count is reached
        while np.count_nonzero(environment) < desired_flower_count:
            # Spawn a random flower
            list_1 = list(flower_information.keys())
            flower_index = random1.randint(0, len(list_1) - 1)
            # Get random index
            x_index = random1.randint(0, environment.shape[0] - 1)
            y_index = random1.randint(0, environment.shape[1] - 1)
            # Ad flower at random location
            environment[x_index, y_index] = list_1[flower_index]
        return environment

    def tournament_selection(rabbits, k=3):
        """
            Description:
                Perform tournament selection with k (=3 by default) opponents
            Arguments:
                rabbits - list, list of rabbit class instances
            Returns:
                champion - class instance
        """
        # Randomly selection one individual, champion by default
        champion_index = randint(0, len(rabbits) - 1)
        for _ in range(0, k):  # k opponents to randomly selected individual
            # Perform tournament
            opponent_index = randint(0, len(rabbits) - 1)
            opponent_fitness = rabbits[opponent_index].get_health()
            champion_fitness = rabbits[champion_index].get_health()
            if opponent_fitness > champion_fitness:
                # The best wins and is stored
                champion_index = opponent_index
        champion = rabbits[champion_index]
        return champion

    def crossover(rabbit1_genotype, rabbit2_genotype, crossover_probability):
        """
            Description:
                Perform crossover from 2 parents to 2 children by r_cross probability, children are copies of
                 parents by default
            Arguments:
                parent1 - list
                parent2 - list
                r_cross - float, in range 0..1
            Returns:
                list_of_children - list, list of lists of children genotype
        """

        # Evaluate recombination
        if random() < crossover_probability:
            # Select crossover point
            crossover_point = randint(1, len(rabbit1_genotype) - 1)
            # Perform crossover
            rabbit_child1_genotype = rabbit1_genotype[:crossover_point] + rabbit2_genotype[crossover_point:]
            rabbit_child2_genotype = rabbit2_genotype[:crossover_point] + rabbit1_genotype[crossover_point:]
        else:
            # Children are copies of parents by default
            rabbit_child1_genotype, rabbit_child2_genotype = rabbit1_genotype, rabbit2_genotype
        list_of_children = [rabbit_child1_genotype, rabbit_child2_genotype]
        return list_of_children

    def mutation(genotype, mutation_probability):
        """
            Description:
                Perform mutation on genotype by mutation_probability probability for each genotype index. genotype
                mutation is replacement by random value drawn from a 0 mean and 2 standard deviation distribution
            Arguments:
                genotype - list
                mutation_probability - float, in range 0..1
            Returns:
                genotype - list
        """

        for i in range(len(genotype)):  # Loop through genotype
            # Evaluate mutation
            if random() < mutation_probability:
                # Replace item with a random value drawn from 0 mean and 1 standard deviation distribution
                if type(genotype[i]) is int:  # Mass or speed
                    genotype[i] += np.random.randn() * 2
                else:  # Memory, i.e., list of training examples
                    pass
        return genotype

    if len(rabbits) == 0:  # Terminate simulation
        print(f'Population size = {len(rabbits)}, program terminates')
        exit()
    for rabbit_index, rabbit in enumerate(rabbits):  # Loop through rabbits
        for _ in range(rabbit.get_speed()):  # Loop for rabbit speed
            # Randomly move rabbit
            rabbit.randomly_move()
            # if flower is at rabbit position, decide to eat or not and take consequence
            if environment[rabbit.get_position()] != 0:
                # Make choice, predict with ANN, eat or not
                flower_index = environment[rabbit.get_position()]
                flower_at_position = flower_information[flower_index].copy()
                flower_size = random1.randint(1, 5)
                flower_at_position['nutrition value'] = \
                    flower_information[flower_index]['nutrition value'][flower_size - 1]
                flower_at_position['flower size'] = flower_size

                if rabbit.decide(flower_at_position):  # If True, decision is eat
                    # Chose to eat
                    # Consequence on of choice on health, i.e. update health
                    nutrition_value = flower_at_position['nutrition value']
                    rabbit.update_health_with_food(nutrition_value)
                    # Consequence of choice on AI, i.e. train ANN with new information
                    if nutrition_value > 0:  # Correct choice to eat
                        rabbit.learn(flower_at_position, y=1)
                    else:  # Incorrect choice to eat
                        rabbit.learn(flower_at_position, y=0)
                    # Update rabbit IQ
                    rabbit.update_iq(flower_information)
                    # Update environment (remove eaten flowers)
                    environment[rabbit.get_position()] = 0  # 0 is vacant ground
                else:
                    # Chose not to eat
                    pass
                # Update health with respect to energy consumed when moving
                rabbit.update_health()
                # If health score is 0, rabbit dies and population decreased by 1
                if rabbit.get_health() <= 0:
                    # Delete rabbit object
                    del rabbits[rabbit_index]  # Delete rabbit
                    break  # Stop moving with this rabbit, i.e., break move loop, and continue to next rabbit

    # Update environment
    environment = update_environment(environment, iteration)
    environment_with_rabbits = environment.copy()  # Create deep copy
    for index, rabbit in enumerate(rabbits):
        # Update environment with rabbits for plotting purposes
        environment_with_rabbits[rabbit.get_position()] = -1

    # Update plot parameters
    list_of_rabbit_iq = []
    list_of_rabbit_mass = []
    list_of_rabbit_speed = []
    list_of_rabbit_health = []
    for rabbit_index, rabbit in enumerate(rabbits):
        rabbit_iq = rabbit.get_iq()
        list_of_rabbit_iq.append(rabbit_iq)

        rabbit_mass = rabbit.get_mass()
        list_of_rabbit_mass.append(rabbit_mass)

        rabbit_speed = rabbit.get_speed()
        list_of_rabbit_speed.append(rabbit_speed)

        rabbit_health = rabbit.get_health()
        list_of_rabbit_health.append(rabbit_health)

    average_rabbit_iq = sum(list_of_rabbit_iq) / len(list_of_rabbit_iq)
    average_rabbit_mass = sum(list_of_rabbit_mass) / len(list_of_rabbit_mass)
    average_rabbit_speed = sum(list_of_rabbit_speed) / len(list_of_rabbit_speed)
    average_rabbit_health = sum(list_of_rabbit_health) / len(list_of_rabbit_health)
    plot_series['Flower population'].append(np.count_nonzero(environment))
    plot_series['Average rabbit IQ'].append(average_rabbit_iq)
    plot_series['Average rabbit mass'].append(average_rabbit_mass)
    plot_series['Average rabbit speed'].append(average_rabbit_speed)

    # Perform breeding
    # Select parents
    selected_parents = []
    for _ in range(rabbit_births_per_generation):
        selected_parent = tournament_selection(rabbits)
        selected_parents.append(selected_parent)
    # Breed children from parents, i.e., create the next generation
    for i in range(0, rabbit_births_per_generation, 2):
        # Selected parents
        rabbit1 = selected_parents[i]
        try:
            rabbit2 = selected_parents[i + 1]
        except IndexError:
            rabbit2 = selected_parents[i - 1]
        rabbit1.update_genotype()
        rabbit2.update_genotype()
        # Crossover
        list_of_children_genome = crossover(rabbit1.get_genotype(), rabbit2.get_genotype(), crossover_probability)
        for child_genome in list_of_children_genome:
            # Mutation
            child_genome = mutation(child_genome, mutation_probability)
            # Add child to population
            child_speed = child_genome[0]
            child_mass = child_genome[1]
            child_list_of_experiences = child_genome[2][0]
            child_list_of_experienced_consequences = child_genome[2][1]
            child_health = average_rabbit_health
            rabbits.append(RabbitClass(grid_size, child_speed, child_mass, child_list_of_experiences,
                                       child_list_of_experienced_consequences, child_health))
            rabbits[-1].update_iq(flower_information)

    # Update one more plot parameter
    plot_series['Rabbit population'].append(len(rabbits))

    iteration += 1  # update iteration counter


grid_size = 100  # Define grid size
iteration = 0  # Initialise iteration counter
rabbits = []  # Initialise list of rabbit class instances
environment = np.zeros((grid_size, grid_size))  # Initialise environment
environment_with_rabbits = environment  # Initialise environment with rabbits, for plotting purposes
# Initialise Evolutionary parameters
rabbit_births_per_generation = 2
mutation_probability = 0.5
crossover_probability = 0.5
plot_series = {'Rabbit population': [],
               'Flower population': [],
               'Average rabbit IQ': [],
               'Average rabbit speed': [],
               'Average rabbit mass': []}
nutrition_scale = 5
flower_information = {}  # Initialise flower information

flower_nutrition_profiles = np.zeros((5, 5))

for i in range(1, 5+1):
    # Calculate flower nutrition for each flower at each size i
    flower_nutrition_profiles[0][i-1] = 1
    flower_nutrition_profiles[1][i-1] = -1
    flower_nutrition_profiles[2][i-1] = ((i - 1) / 9) * 2 - 1
    flower_nutrition_profiles[3][i-1] = ((-i + 1) / 9) * 2 + 1
    flower_nutrition_profiles[4][i-1] = np.cos(i)

nutrition_scale = 5
flower_nutrition_profiles = flower_nutrition_profiles * nutrition_scale  # Scale flower nutrition

# Construct flower information dict, with flower numbering, color, and nutritional size relationship
flower_information = {1: {'color': 'yellow', 'nutrition value': flower_nutrition_profiles[0][:]},
                      2: {'color': 'red',    'nutrition value': flower_nutrition_profiles[1][:]},
                      3: {'color': 'blue',   'nutrition value': flower_nutrition_profiles[2][:]},
                      4: {'color': 'green',  'nutrition value': flower_nutrition_profiles[3][:]},
                      5: {'color': 'black',  'nutrition value': flower_nutrition_profiles[4][:]}}

# Start simulator
pycxsimulator.GUI().start(func=[initialize, observe, update])
