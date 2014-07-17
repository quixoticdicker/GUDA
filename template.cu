#define MEMBERS

#define SELECT_MUTATION 1

#include "guda.h"

// put any defines or anything in constant memory here

/**
 * Initialization for your individuals.
 * You may treat this similarly to a constructor.
 * It should take no arguments, however, you may
 * use information in constant memory.
 */
__device__ void Individual::init()
{
}

/**
 * Evaluate your individuals.
 * This function will return a float indicating
 * an individual's fitness. A High fitness is better.
 * Like in nature, a high fitness implies that the
 * individual will live longer and reproduce.
 */
__device__ float Individual::evaluate()
{
    float fitness = 0.0f;
	// your code
    return fitness;
}

/**
 * Mutate your individuals.
 * Change the "DNA" of your individuals slightly.
 * This usually consists of randomly altering bits
 * or changing a value with a Gaussian etc. Be careful
 * not to alter the value too much or you'll just end
 * up with another random individual.
 */
__device__ void Individual::mutate()
{
}

/**
 * Print out the important traits of the best individual.
 */
void print_individual(Individual indiv)
{
}

/**
 * This is your main function to alter for your own purpose.
 * Run will return the best individual. Run takes two arguments.
 * The first one is the number of generations and the second
 * is the chunk size. The chunk size might need to be played
 * with a bit. It is important because a gpu function will fail
 * if it takes two long to execute. This means that if we try
 * to do too many generations in one gpu function, the call will
 * fail. The chunk size is the number of generations that will
 * execute in each function call. If you're getting errors, try
 * a smaller chunk size.
 */
int main(int argc, char *argv[])
{
    Individual best = run(atoi(argv[1]), atoi(argv[2]));
    printf("The best individual found is\n");
    print_individual(best);
    printf("and it has fitness %f.\n", best.getFitness());
    return 0;
}
