__device__ float pharaohRand()
{
    int seed = clock();
    curandState_t myState;
    curand_init(seed, blockIdx.x * blockDim.x, threadIdx.x, &myState);
	
    return (float) curand_uniform(&myState);
}
