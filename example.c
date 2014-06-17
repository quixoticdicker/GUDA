#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*

This implements an overly simplistic genetic algorithm for finding the
square root of 2. The point is to see how we might implement a genetic
algorithm on a GPU. The for loop in generation() would be replaced by
a kernel call.

Compile: gcc -std=c99 example.c
Run: ./a.out

*/

// CUSTOMIZABLE PART

typedef struct {
    double value;
} indiv_t;

void init(indiv_t *indiv) {
    indiv->value = 2 * (double) rand() / (double) RAND_MAX;
}

double fitness(indiv_t indiv) {
    return -fabs(2.0 - indiv.value * indiv.value);
}

indiv_t mutate(indiv_t indiv) {
    // random # in [0.9, 1.1)
    double r = ((1.1-0.9) * (double) rand() / (double) RAND_MAX + 0.9);
    indiv_t mutated = indiv;
    mutated.value *= r;
    return mutated;
}

// COMMON PART

// utility
double avg_fitness(indiv_t *pop, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
	sum += fitness(pop[i]);
    return sum / n;
}

//
double avg_value(indiv_t *pop, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
	sum += pop[i].value;
    return sum / n;
}

int main() {
    int n = 100000;
    srand(time(0));

    indiv_t old_pop[n];
    indiv_t new_pop[n];

    // initialize the populations
    for (int i = 0; i < n; i++) {
	init(&old_pop[i]);
	new_pop[i] = old_pop[i];
    }

    printf("%f\n", avg_fitness(new_pop, n));
    printf("%f\n", avg_value(new_pop, n));

    for (int g = 0; g < 1000; g++) {

	// compute the fitness of the old population
	double fit[n];
	for (int i = 0; i < n; i++) {
	    fit[i] = fitness(old_pop[i]);
	}

	// tournament selection: 
	//   pick two random individuals from my old population
	//   put the best one in my location

	for (int i = 0; i < n; i++) {
	    int indiv1 = rand() % n;
	    int indiv2 = rand() % n;

	    if (fit[indiv1] > fit[indiv2])
		new_pop[i] = old_pop[indiv1];
	    else
		new_pop[i] = old_pop[indiv2];
	}

	for (int i = 0; i < n; i++) {
	    old_pop[i] = mutate(new_pop[i]);
	}
    }

    printf("%f\n", avg_fitness(new_pop, n));
    printf("%f\n", avg_value(new_pop, n));

    return 0;
}
