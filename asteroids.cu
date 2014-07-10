#define MEMBERS float theta; float x, y;

#define NUM_ROIDS 5
__constant__ float roids_x[] = {-80.0f, 10.0f, 41.0f, 81.0f, 96.0f};
__constant__ float roids_y[] = {-80.0f, 10.0f, 41.0f, 81.0f, 96.0f};
__constant__ float roids_r[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

#define SELECT_MUTATION 

#include "guda.h"

__device__ void Individual::init() {	
    x = pharaohRand() * 200.0f - 100.0f;
    y = pharaohRand() * 200.0f - 100.0f;
    theta = pharaohRand() * 2 * M_PI;
}

__device__ float Individual::evaluate()
{
    float m, b, mp, bp, xa, ya, xs, ys, d, xd, yd;
    int i;
    float fitness = 0.0f;
    m = sinf(theta) / cosf(theta);
    b = y - m * x;
    mp = -1.0f / m;
    
    for (i = 0; i < NUM_ROIDS; i++)
    {
	xa = roids_x[i];
	ya = roids_y[i];
	bp = ya - mp * xa;
	
	xs = (bp - b)/(m - mp);
	ys = m * xs + b;
	
	xd = xa - xs;
	yd = ya - ys;
	d = sqrtf(xd * xd + yd * yd);
	
	if (d < roids_r[i]) fitness++;
    }
    return fitness;

    /*
    // xMutagen, yMutagen, mMutagen
    int xMutagen = 0;
    int yMutagen = 0;
    int thetaMutagen = 0;
    int i;
    for(i = 0; i < sizeof(float) * 8; i++)
    {
	if(pharaohRand() < RATE)
	{
	    xMutagen++;
	}
	if(pharaohRand() < RATE)
	{
	    yMutagen++;
	}
	if(pharaohRand() < RATE)
	{
	    thetaMutagen++;
	}
	xMutagen <<= 1;
	yMutagen <<= 1;
	thetaMutagen <<= 1;
    }
    i = * (int *) &x; // evil floating point bit level hacking
    i ^= xMutagen;
    x = * (float *) &i;
    
    i = * (int *) &y;
    i ^= yMutagen;
    y = * (float *) &i;
    
    i = * (int *) &theta;
    i ^= thetaMutagen;
    theta = * (float *) &i;
    */
}

__device__ void Individual::mutate()
{
    x += 10*cleopatra();
    y += 10*cleopatra();
    theta += 0.1745 * cleopatra();
}

void print_individual(Individual indiv)
{
    printf("x = %f, y = %f, theta = %f\n", indiv.x, indiv.y, indiv.theta);
}

int main()
{
    Individual best = run();
    printf("The best individual found is\n");
    print_individual(best);
    printf("and it has fitness %f.\n", best.getFitness());
    return 0;
}
