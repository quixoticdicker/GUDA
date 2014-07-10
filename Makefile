all: prototype.out asteroids.out

prototype: prototype.out

asteroids: asteroids.out

prototype.out: prototype.cu pharaohrand.h guda.h
	nvcc -arch=sm_21 prototype.cu -o prototype.out

asteroids.out: asteroids.cu pharaohrand.h guda.h
	nvcc -arch=sm_21 asteroids.cu -o asteroids.out

clean:
	rm -rf a.out
