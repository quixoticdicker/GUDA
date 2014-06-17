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

void generation(indiv_t *pop, int n) {
    for (int i = 0; i < n; i++) {
	// maybe replaced by child
	indiv_t child = mutate(pop[i]);
	if (fitness(child) > fitness(pop[i]))
	    pop[i] = child;
    }
}

// utility
double avg_fitness(indiv_t *pop, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
	sum += fitness(pop[i]);
    return sum / n;
}

int main() {
    srand(time(0));

    // initialize the population
    int n = 10;
    indiv_t pop[n];
    for (int i = 0; i < n; i++)
	init(&pop[i]);

    printf("%f\n", avg_fitness(pop, n));

    // generations
    for (int g = 0; g < 1000; g++)
	generation(pop, n);

    printf("%f\n", avg_fitness(pop, n));

    return 0;
}
