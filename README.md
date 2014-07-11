GUDA: Genetic Algorithm Library on CUDA
=======================================

Thank you for showing interest in GUDA. GUDA is a project that was started to both educate computer science students about parallel programming in CUDA and to assist people with no experience in CUDA in utilizing their NVIDIA graphics card for genetic algorithm purposes.

### Background
To begin, let me give you some background on genetic algorithms and CUDA. If you feel comfortable with these concepts, feel free to skip to the next section. Genetic algorithms are a class of algorithm that takes advantage of a process similar to the evolution that we see in nature. Just as a reminder, in nature we see populations evolve due to survival of the fittest. Individuals of this population gain traits from their parents or mutations that occur in their DNA. If they are fitter than others in the population, they will continue to live and reproduce. On a computer, we will define individuals of the population, a function for combining individuals (known as **crossover**), a function for **mutating** individuals, and a function for **evaluating** individuals. The function that evaluates individuals will assign a fitness to each individual. There are then several ways to choose the new population based on the previous one. One way is to choose the n fittest members of the population, copy them, and mutate the copies until we fill up the population again. Unfortunately this, and many other methods, require a sequential algorithm; it is for this reason that GUDA utilizes a method called tournament selection. Tournament selection goes through each member of the new population, picks two members from the old population randomly, and assigns the one with the higher fitness to have that spot in the new population. This doesn't work perfectly since the random members that are chosen could always be from the poorer half of the population. It is also possible for the new population to consist entirely of one individual. Even so, tournament selection works well for large populations and allows us to do the selection step in parallel.

CUDA is a language developed by NVIDIA that we're using in conjunction with C++ that allows programmers to utilize their graphics cards for general purpose programming. Graphics cards have hundreds or thousands of cores which is much greater than the 4 - 12 cores that you might find on the nicest CPU's. The cores on a GPU are organized onto streaming multiprocessors (SM's). NVIDIA GPU's mostly have cores that can do single precision operations.

### Utilization
Although most of it is already done for you, we are going to ask you to define a few things. The first thing we will ask you to define is the variables that the Individual will have. You should see the following code in template.cu:
```C++
#define MEMBERS
```
After this, write all of the variables that you would like the individual to have. For example, say were using a genetic algorithm to find the square root of two. We would want our individual to store a value corresponding with that individual's guess for the square root of two:
```C++
#define MEMBERS float sqrtTwo;
```
You'll noticed that I defined a float instead of a double. When using CUDA, **it is often faster to use floats** since NVIDIA graphics cards mostly have cores that can do single precision operations. Next, we need you to define a function to initialize an individual. You will see the following code:
```C++
__device__ void init()
{

}
```
Replace this with something that initializes your variables. For example:
```C++
__device__ void init()
{
	sqrtTwo = pharaohRand() * 100.0f - 50.0f;
}
```
This generates a random float between -50 and 50 and puts it in sqrtTwo. The first thing you'll notice is this pharaohRand() function. This is a function that we're giving you that generates a random float between 0 and 1. We're scaling up what is returned so we get a number between 0 and 100. Finally, we're subtracting 50 so we get a number from -50 to 50. The next thing you'll notice is that we are writing 100.0f and 50.0f instead of just 100 and 50. This is to ensure that the compiler treats these numbers as floats.

Next, define the mutate function. Find the following code:
```C++
__device__ void mutate()
{

}
```
One common way to mutate individuals is to flip random bits of the individual's DNA according to some mutation rate. You can do this by generating a random int or some variable that is the size of the variable you're mutating then using the XOR operator (^) to flip those bits. This might look like the following:
```C++
__device__ void mutate()
{
	int mutagen = 0;
	int i;
	float mutationRate = 0.03f;
	for(i = 0; i < sizeof(float) * 8; i++)
	{
		if(pharaohRand() < mutationRate) // only flips 3% of bits
		{
			mutagen++;
		}
		mutagen <<= 1;
	}
	i = *(int*) &sqrtTwo;   // parse sqrtTwo as an int
	i ^= mutagen;           // do xor on i (sqrtTwo parsed as int)
	sqrtTwo = *(float*) &i; // store this value back into sqrtTwo
}
```
This code is a) very confusing and b) sometimes dangerous. Since there are a range of numbers that are considered not a number (nan), we may accidentally find one of these instead of just changing the value. We encourage you, instead, to alter your variables with a normal distribution. This would look like:
```C++
__device__ void mutate()
{
	sqrtTwo += cleopatra() * 5.0f;
}
```
This will add a random number to sqrtTwo that is centered at 0 and has a standard deviation of 5. You'll notice this cleopatra() function. This is another function that we provide that returns a normally distributed float that is centered at 0 and has a standard deviation of 1. This code is much simpler, less prone to error, and should provide similar results.

Finally, we need you to write an evaluate function that will return the fitness of the individual. Remember that individuals that are closer to the desired result should have a higher fitness. Find the code below:
```C++
__device__ float evaluate()
{
	
}
```
For our example, we want it to return a higher value if the number is closer to the square root of two.
```C++
__device__ float evaluate()
{
	return -fabsf(sqrtTwo * sqrtTwo - 2);
}
```
We know that our guess, when multiplied with itself, should be two. We find the magnitude of the difference, and take the negative to ensure that a smaller difference corresponds with a higher fitness. For example, if sqrtTwo actually equals the square root of two, the function would return 0 and if sqrtTwo was 4, the function would return -14 (-|4 * 4 - 2| = -|16 - 2| = - |14| = -14). You'll notice that we're using the fabsf() function which is a function in CUDA's math library. You can find all math functions that are usable in CUDA [here](http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE).

We will still rely on you to write your own main function. In your main function you just call the function run() which returns a pointer to the best Individual.

There are some other parameters that you can play with to try to get the most out of your custom genetic algorithm. For example, you can choose things like the population size and the number of generations. Because of the architecture of NVIDIA graphics cards, we split the population into islands. You can define a number of islands and a population per island. For example, say we wanted 7 islands, 100 individuals per island, and 200 generations; this would look like:
```C++
#define POP_PER_ISLAND 100
#define NUM_ISLANDS 7
#define GENERATIONS 200
```
We also have some features that can be used to speed up the process. One of these is the SELECT_MUTATION feature. This feature only allows a mutation to occur if the resulting individual has a higher fitness. To enable this feature, define SELECT_MUTATION.
```C++
#define SELECT_MUTATION
```

There may also be some times when all of your threads access the same data. For example, look at [this problem](https://icpcarchive.ecs.baylor.edu/index.php?option=com_onlinejudge&Itemid=8&category=622&page=show_problem&problem=4532) from the 2014 ICPC. We implement a genetic algorithm solution to this problem where the members of the population are possible beams defined by a point and a slope. To evaluate a beam, a thread needs to access the locations and radii of each asteroid. We are able to do this using constant memory. We define constant memory before the include "guda.h". This would look like:
```C++
#define NUM_ROIDS 5
__constant__ float roids_x[] = {-80.0f, 10.0f, 41.0f, 81.0f, 96.0f};
__constant__ float roids_y[] = {-80.0f, 10.0f, 41.0f, 81.0f, 96.0f};
__constant__ float roids_r[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

// some other defines etc.

#include "guda.h"
```
Here we have defined five asteroids along the line y = x, all with a radius of 1.0. You can then easily access this data from your evaluate function just as if it were a global variable.
```C++
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
```