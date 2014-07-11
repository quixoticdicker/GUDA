#define MEMBERS float theta; float x, y;

#define SELECT_MUTATION 1

#include "guda.h"

#define NUM_ROIDS 50
__constant__ float roids_x[] = {
    938.7938849063594f,
    -464.5763836813552f,
    19.432404541948472f,
    -244.50853851300053f,
    600.953861394762f,
    -516.1737023123312f,
    915.3211624518842f,
    694.341350160523f,
    361.616373321252f,
    439.7162088153916f,
    -399.10355252656893f,
    -543.0104614896177f,
    -599.6850129929339f,
    656.27496636919f,
    13.759342443270725f,
    -614.191660932717f,
    946.3552940894917f,
    39.39311699429891f,
    -407.388409319698f,
    779.9285877744551f,
    -690.2865817871857f,
    728.0322967712957f,
    98.71656128218319f,
    900.6651536324227f,
    -568.3190715412222f,
    -552.6975403808376f,
    103.98470426638164f,
    -318.3700407068151f,
    -596.1427310774898f,
    -414.7174488285734f,
    642.2051304578029f,
    447.18219212971735f,
    -441.95768038300764f,
    -123.88377322819167f,
    -762.5816987775888f,
    -20.703398631566756f,
    564.940912022616f,
    -257.01913474646494f,
    90.20469831495552f,
    573.8832191273318f,
    -254.2262348104747f,
    40.80840293707092f,
    -853.4148337294014f,
    374.31850989031864f,
    232.95291269793006f,
    42.56502509333495f,
    -617.0400085952213f,
    706.9649264836048f,
    -689.7674365377284f,
    929.9446355018665f
};

__constant__ float roids_y[] = {
    938.7938849063594f,
    -464.5763836813552f,
    19.432404541948472f,
    -244.50853851300053f,
    600.953861394762f,
    -516.1737023123312f,
    915.3211624518842f,
    694.341350160523f,
    361.616373321252f,
    439.7162088153916f,
    -399.10355252656893f,
    -543.0104614896177f,
    -599.6850129929339f,
    656.27496636919f,
    13.759342443270725f,
    -614.191660932717f,
    946.3552940894917f,
    39.39311699429891f,
    -407.388409319698f,
    779.9285877744551f,
    -690.2865817871857f,
    728.0322967712957f,
    98.71656128218319f,
    900.6651536324227f,
    -568.3190715412222f,
    -552.6975403808376f,
    103.98470426638164f,
    -318.3700407068151f,
    -596.1427310774898f,
    -414.7174488285734f,
    642.2051304578029f,
    447.18219212971735f,
    -441.95768038300764f,
    -123.88377322819167f,
    -762.5816987775888f,
    -20.703398631566756f,
    564.940912022616f,
    -257.01913474646494f,
    90.20469831495552f,
    573.8832191273318f,
    -254.2262348104747f,
    40.80840293707092f,
    -853.4148337294014f,
    374.31850989031864f,
    232.95291269793006f,
    42.56502509333495f,
    -617.0400085952213f,
    706.9649264836048f,
    -689.7674365377284f,
    929.9446355018665f
};

__constant__ float roids_r[] = {
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
};

__device__ void Individual::init() {	
    x = pharaohRand() * 2000.0f - 1000.0f;
    y = pharaohRand() * 2000.0f - 1000.0f;
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
    x += 100*cleopatra();
    y += 100*cleopatra();
    theta += 0.1745 * cleopatra();
}

void print_individual(Individual indiv)
{
    printf("x = %f, y = %f, theta = %f\n", indiv.x, indiv.y, indiv.theta);
}

int main(int argc, char *argv[])
{
    Individual best = run(atoi(argv[1]), atoi(argv[2]));
    printf("The best individual found is\n");
    print_individual(best);
    printf("and it has fitness %f.\n", best.getFitness());
    return 0;
}
