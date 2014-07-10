GUDA: Genetic Algorithm Library on CUDA
=======================================

Thank you for showing interest in GUDA. GUDA is a project that was started to both educate computer science students about parallel programming in CUDA and to assist people with no experience in CUDA in utilizing their NVIDIA graphics card for genetic algorithm purposes.

### Background
To begin, let me give you some background on genetic algorithms and CUDA. If you feel comfortable with these concepts, feel free to skip to the next section. Genetic algorithms are a class of algorithm that takes advantage of a process similar to the evolution that we see in nature. Just as a reminder, in nature we see populations evolve due to survival of the fittest. Individuals of this population gain traits from their parents or mutations that occur in their DNA. If they are fitter than others in the population, they will continue to live and reproduce. On a computer, we will define individuals of the population, a function for combining individuals (known as crossover), a function for mutating individuals, and a function for evaluating individuals. The function that evaluates individuals will assign a fitness to each individual. There are then several ways to choose the new population based on the previous one. One way is to choose the n fittest members of the population, copy them, and mutate the copies until we fill up the population again. Unfortunately this, and many other methods, require a sequential algorithm; it is for this reason that GUDA utilizes a method called tournament selection. Tournament selection goes through each member of the new population, picks two members from the old population randomly, and assigns the one with the higher fitness to have that spot in the new population. This doesn't work perfectly since the random members that are chosen could always be from the poorer half of the population. It is also possible for the new population to consist entirely of one individual. Even so, tournament selection works well for large populations and allows us to do the selection step in parallel.

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
__device__ void init()
{
	sqrtTwo = pharaohRand() * 100.0f - 50.0f;
}
This generates a random float between -50 and 50 and puts it in sqrtTwo. The first thing you'll notice is this pharaohRand() function. This is a function that we're giving you that generates a random float between 0 and 1. We're scaling up what is returned so we get a number between 0 and 100. Finally, we're subtracting 50 so we get a number from -50 to 50. The next thing you'll notice is that we are writing 100.0f and 50.0f instead of just 100 and 50. This is to ensure that the compiler treats these numbers as floats.

Next