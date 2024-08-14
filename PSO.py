import operator
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools
from deap import creator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Read data
database = pd.read_csv('..\database.csv', encoding='GB2312')
inputdata = pd.read_csv('..\inputdata.csv', encoding='GB2312')
TSCM = [[1, 0.7, 0.3, 0.2, 0],
        [0.7, 1, 0.4, 0.3, 0],
        [0.3, 0.4, 1, 0.8, 0],
        [0.2, 0.3, 0.8, 1, 0],
        [0, 0, 0, 0, 1]]
decision_variables = 9
row_names = ['碎石土', '砂土', '粉土', '粘性土', '特殊性土']
col_names = ['碎石土', '砂土', '粉土', '粘性土', '特殊性土']
matrix = pd.DataFrame(index=row_names, columns=col_names)
for i in range(len(row_names)):
    for j in range(len(col_names)):
        matrix[col_names[j]][i] = TSCM[i][j]


# Text-type data
def calculate_text(text1, text2):
    # Create CountVectorizer object to convert text to word frequency vector
    texts = [text1, text2]
    vectorizer = CountVectorizer()
    vectorized_texts = vectorizer.fit_transform(texts)
    # Calculate the similarity between two texts
    def bag_of_words_similarity(text1, text2):
        vector1 = vectorizer.transform([text1])
        vector2 = vectorizer.transform([text2])
        similarity = vector1.dot(vector2.T).toarray()[0][0] / np.linalg.norm(vector1.toarray()[0]) / np.linalg.norm(
            vector2.toarray()[0])
        return similarity
    return bag_of_words_similarity(texts[0], texts[1])

# Similarity matrix data
def calculate_juzhen(type1, type2):
    return matrix[type1][type2]

# Interval-type data
def calculate_qujian(upper1, lower1, upper2, lower2):
    upper_jiaoji = min(upper1, upper2)
    lower_jiaoji = max(lower1, lower2)
    upper_bingji = max(upper1, upper2)
    lower_bingji = min(lower1, lower2)
    if upper_jiaoji < lower_jiaoji:
        return 0
    return (upper_jiaoji - lower_jiaoji) / (upper_bingji - lower_bingji)

# Float-type data
def calculate_num(x, y, lmax, lmin):
    return 1 - abs(x-y) / (lmax-lmin)

# Categorical-type data
def calculate_fuzhi(x, y):
    return 1 - abs(x-y) / 6

# Generate n random numbers with sum 1
def generate_random_numbers(n, target_sum):
    numbers = []
    while len(numbers) < n-1:
        num = random.uniform(0, target_sum)
        if sum(numbers) + num > target_sum:
            num = random.uniform(0, target_sum - sum(numbers))
        numbers.append(num)
    numbers.append(1-sum(numbers))
    return numbers

# Particle Generator
def generate(size, smin, smax):
    part = creator.Particle(generate_random_numbers(decision_variables, 1))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

# To update velocity and position of particles
def updateParticle(part, best, w1, c1, c2):
    w1_array = (random.uniform(0, w1) for _ in range(len(part)))  # w1 coeff
    c1_array = (random.uniform(0, c1) for _ in range(len(part)))  # c1 coeff
    c2_array = (random.uniform(0, c2) for _ in range(len(part)))  # c2 coeff

    # calculating velocity term = inertia * current speed
    inertia_term = map(operator.mul, w1_array, part.speed)

    # cognitive term = c1 * (particle best - particle current)
    cognitive_term = map(operator.mul, c1_array, map(operator.sub, part.best, part))

    # social term = c2 * (pop best - pop current)
    social_term = map(operator.mul, c2_array, map(operator.sub, best, part))

    # velocity update
    part.speed = list(map(operator.add, inertia_term, map(operator.add, cognitive_term, social_term)))
    # Check if any new position is negative and set it to 0 if true
    new_positions = list(map(operator.add, part, part.speed))  # Convert map to list first
    for i in range(len(part)):
        if new_positions[i] < 0:
            new_positions[i] = 0

    part[:] = new_positions  # Update the positions of the particle


# Fitness function
def fitness(individual):
    loss = 0
    for new in range(len(inputdata)):
        max_openrate = 0
        max_simi = 0
        for basis in range(len(database)):
            simi = []
            simi.append(calculate_text(inputdata['Geotechnical type'][new], database['Geotechnical type'][basis]))
            simi.append(calculate_juzhen(inputdata['Typical soils'][new], database['Typical soils'][basis]))
            simi.append(calculate_text(inputdata['Unfavourable geology'][new], database['Unfavourable geology'][basis]))
            simi.append(
                calculate_qujian(inputdata['UCSupper'][new], inputdata['UCSlower'][new], database['UCSupper'][basis],
                                 database['UCSlower'][basis]))
            simi.append(calculate_fuzhi(inputdata['Permeation'][new], database['Permeation'][basis]))
            simi.append(
                calculate_num(inputdata['Maximum groundwater level'][new], database['Maximum groundwater level'][basis],
                              max(database['Maximum groundwater level']), min(database['Maximum groundwater level'])))
            simi.append(calculate_qujian(inputdata['Burial depth upper'][new], inputdata['Burial depth lower'][new],
                                         database['Burial depth upper'][basis], database['Burial depth lower'][basis]))
            simi.append(calculate_num(inputdata['Length'][new], database['Length'][basis], max(database['Length']),
                                      min(database['Length'])))
            simi.append(calculate_num(inputdata['Calibre'][new], database['Calibre'][basis], max(database['Calibre']),
                                      min(database['Calibre'])))
            new_simi = np.where(np.isnan(simi), 0, simi)
            sum_simi = individual[0] * new_simi[0] + individual[1] * new_simi[1] + individual[2] * new_simi[2] + \
                       individual[3] * new_simi[3] + individual[4] * new_simi[4] + \
                       individual[5] * new_simi[5] + individual[6] * new_simi[6] + individual[7] * new_simi[7] + \
                       individual[8] * new_simi[8]
            if sum_simi > max_simi:
                max_simi = sum_simi
                max_openrate = database['Open rate'][basis]
        loss += round(abs(max_openrate - inputdata['Open rate'][new]), 6)
    return loss,

# To plot performance graph
def plot_fit(best, gen_max):
    plt.plot(range(0, gen_max + 1, 10), best)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()


# Building multi-objective optimization problems
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, best=None)
toolbox = base.Toolbox()
toolbox.register("particle", generate, size=decision_variables, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("evaluate", fitness)
toolbox.register("update", updateParticle)

def main():
    # Initialize population
    pop = toolbox.population(n=50)
    gen_max = 20
    best = None
    best_fit = []
    best_decision = []
    w1 = 0.72984
    c1 = c2 = 1.496180
    for i in range(gen_max + 1):
        # velocity damping using inertia over generations
        w1 = 0.72984 - (i / (gen_max * 2))
        # Adjusting coe7ff to increase local exploration and decrease global exploration over generations
        c1 = 1.496180 + (i / gen_max)
        c2 = 1.496180 - (i / gen_max)
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            # Update P_best
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            # Update G_best
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        # Update velocity and position
        for part in pop:
            toolbox.update(part, best, w1, c1, c2)

        # if i % 10 == 0:
        best_fit.append(best.fitness.values)
        best_decision.append(best)
        print("Generation:{}   Fitness:{}".format(i, best_fit[-1]))


    outmulti = pd.DataFrame(data={'fit': best_fit,
                                  'dicision': best_decision})
    outmulti.to_csv('..\output.csv')


if __name__ == "__main__":
    main()
