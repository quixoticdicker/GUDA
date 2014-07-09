__device__ float pharaohRand()
{
    int seed = clock();
    curandState_t myState;
    curand_init(seed, blockIdx.x * blockDim.x, threadIdx.x, &myState);
	
    return (float) curand_uniform(&myState);
}

__device__ float cleopatra()
{
	float v1, v2, s;
	do {
		v1 = 2.0f * pharaohRand() - 1.0f;
		v2 = 2.0f * pharaohRand() - 1.0f;
		s = v1 * v1 + v2 * v2;
	} while (s >= 1.0f || s == 0.0f);
	float multiplier = sqrtf(-2.0f * logf(s)/s);
	return v1 * multiplier;
}
