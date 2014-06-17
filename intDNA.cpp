#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct Individual {
	int DNA;
};

Individual crossover(Individual mom, Individual dad)
{
    int pivot = rand() % (sizeof(int) * 8);
	Individual child;
	int partMom = mom.DNA & (-1 << pivot);
	int partDad = dad.DNA & (-1 << pivot);
	child.DNA = partMom + (dad.DNA - partDad);
}

double fitness(Individual individual)
{
    return individual.DNA;
}

Individual mutate(Individual individual)
{
    int i;
    int mutagen = 0;
    int mutationRate = 3;
    for(i = 0; i < sizeof(int) * 8; i++)
    {
        if(rand() % 100 < mutationRate)
        {
            i += 1;
        }
        i << 1;
    }
    
    individual.DNA ^= i;
}

int main(void)
{
	printf("hello world!");
}
