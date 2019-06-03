import random
import numpy as np
import matplotlib.pyplot as plt
import math

#CONSTANTS
# tamPopulation
# cruzamento
# coeficienteDeVariacaoDiferencial

results = []

bestList = [] # graphic

def DifferentialEvolution(limitatoins, mutation, crossover, populationSize, interactions, phi, beta, sigma):
    print('initiantion DE...')
    def fnFitness(phi, beta, sigma):
        print("PHI --> ", phi)
        print("Beta --> ", beta)
        print("Sigma --> ", sigma)
        f = np.zeros((len(phi), len(phi[0][0])))
        for ell in range(0, len(phi[0][0])):
            for k in range(0, len(phi)):
                f[k, ell] = beta[k, ell, ell]
                deno = 0
                for j in range(1, len(phi[0][0])):
                    for kline in range(0, len(phi)):
                        if j != ell:
                            # if k > ell:
                            deno += np.inner(phi[k, :, ell], phi[kline, :, j]) * beta[kline, j, ell]
                deno += sigma
                f[k, ell] /= deno
            # self.fitnessNote = np.sum(f)
            # print("fitnessNote = ", abs(np.sum(f)))
            return abs(np.sum(f))

    dimentions = len(phi)
    population = np.random.rand(populationSize, dimentions)
    minLimite, maxLimite = np.asarray(limitatoins).T
    print("Limites ---> %s, %s " % (minLimite, maxLimite))
    diff = np.fabs(minLimite - maxLimite)
    print("diff -> ", diff)
    pop_denom = minLimite + population * diff

    pop = np.zeros((len(beta), len(beta), len(beta[0]), int(populationSize)))
    # for j in range(populationSize):
    fitness = fnFitness(pop[:, :, :], beta, sigma)
    # fitness = 5
    print("fitness --> ", fitness)
    best_idx = np.argmax(fitness)
    best = pop_denom[best_idx]

    for i in range(interactions):
        for j in range(populationSize):
            idxs = [idx for idx in range(populationSize) if idx != j]
            a,b,c = population[np.random.choice(idxs, 3, replace=False)]
            mutated = np.clip(a + mutation * (b-c), 0,1)
            crossPoints = np.random.rand(dimentions) < crossover
            if not np.any(crossPoints):
                crossPoints[np.random.randint(0, dimentions)] = True
            trial = np.where(crossPoints, mutated, population[j])
            trial_donorm = minLimite + trial * diff

            print("trialDenorm --> ", trial_donorm)

            f = 3

            if f < fitness[j]:
                fitness[j] = f
                population[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_donorm
        print("Best --> ", best)
        print("Best Fitness --> ", fitness[best_idx])
