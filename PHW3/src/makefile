CC = gcc

all: p1a p1b p2 p3

p1a: p1a.c
	$(CC) -lrt -fopenmp -o p1a p1a.c

p1b: p1b.c
	$(CC) -lrt -fopenmp -o p1b p1b.c

p2: p2.c
	$(CC) -lrt -fopenmp -o p2 p2.c

p3: p3.c
	$(CC) -lrt -lpthread -o p3 p3.c

clean:
	rm p1a p1b p2 p3

