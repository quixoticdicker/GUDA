all: prototype.out

prototype.out: prototype.cu pharaohrand.h
	nvcc -arch=sm_21 prototype.cu -o prototype.out

clean:
	rm -rf a.out
