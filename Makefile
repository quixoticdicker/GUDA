all: prototype.out

prototype: prototype.cu pharaohrand.h
	nvcc -arch=sm_21 prototype.cu -o prototype.out

clean:
	rm -rf a.out
