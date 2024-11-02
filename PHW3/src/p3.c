#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define h  800 
#define w  800

#define input_file  "input.raw"
#define output_file "output.raw"

#define NUM_CLUSTERS 6

int NUM_THREADS;
unsigned char *a;  // the image
int cluster[NUM_CLUSTERS] = {0, 65, 100, 125, 190, 255};  // initial mean values
int global_sum[NUM_CLUSTERS], global_count[NUM_CLUSTERS];  // shared among threads
pthread_mutex_t mutex[NUM_CLUSTERS];  // one mutex for each cluster
pthread_mutex_t r_mutex;
pthread_cond_t cond_var;
int r=0;

void *threadFunc(void *arg) {
    int id = (intptr_t)arg;
    int start = (h * w / NUM_THREADS) * id;  // start index 
    int end = (h * w / NUM_THREADS) * (id + 1);  // end index 

    for (int iteration = 0; iteration < 50; iteration++) {
        int local_sum[NUM_CLUSTERS] = {0};
        int local_count[NUM_CLUSTERS] = {0};
        
        // Assign each data element to the closest cluster and update sum and count
        for (int i = start; i < end; i++) {
            int min_distance = 255;
            int cluster_index = 0;
            for (int j = 0; j < NUM_CLUSTERS; j++) { // finding the closest cluster
                int distance = abs(a[i] - cluster[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    cluster_index = j;
                }
            }
            local_sum[cluster_index] += a[i];
            local_count[cluster_index]++;
        }

        // Combine local sums and counts to global ones
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            pthread_mutex_lock(&mutex[i]);
            global_sum[i] += local_sum[i];
            global_count[i] += local_count[i];
            pthread_mutex_unlock(&mutex[i]);
        }

        // Check the value of r and act accordingly
        pthread_mutex_lock(&r_mutex);
        if (r < NUM_THREADS - 1) {
            r++;
            pthread_cond_wait(&cond_var, &r_mutex);
        } else {
            r = 0;

            // Recompute the mean value of each cluster
            for (int i = 0; i < NUM_CLUSTERS; i++) {
                if (global_count[i] != 0) {
                    cluster[i] = global_sum[i] / global_count[i];
                }
                global_sum[i] = 0;  // Reset for the next iteration
                global_count[i] = 0;  // Reset for the next iteration
            }

            pthread_cond_broadcast(&cond_var);
        }
        pthread_mutex_unlock(&r_mutex);
    }
    return NULL;
}


int main(int argc, char** argv){
    int i, j, k;
    FILE *fp;
	struct timespec start, stop; 
	double time;
    NUM_THREADS = atoi(argv[1]); // size of thread

  	a = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
    
	// the matrix is stored in a linear array in row major fashion
	if (!(fp=fopen(input_file, "rb"))) {
		printf("can not opern file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), w*h, fp);
	fclose(fp);
    
	// measure the start time here
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

    // initialize mutexes
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        pthread_mutex_init(&mutex[i], NULL);
    }
    pthread_mutex_init(&r_mutex, NULL);
    pthread_cond_init(&cond_var, NULL);

    // create threads and run
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        int rc = pthread_create(&threads[i], NULL, threadFunc, (void*)(intptr_t)i);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // synchronize the threads after all iterations
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // After all iterations, perform the final assignment of each data element to its cluster's mean
    for (int i = 0; i < h * w; i++) {
        int min_distance = 255;
        int cluster_index = 0;
        for (int j = 0; j < NUM_CLUSTERS; j++) {
            int distance = abs(a[i] - cluster[j]);
            if (distance < min_distance) {
                min_distance = distance;
                cluster_index = j;
            }
        }
        a[i] = cluster[cluster_index];
    }

    // destroy the mutexes and condition variable
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        pthread_mutex_destroy(&mutex[i]);
    }
    pthread_mutex_destroy(&r_mutex);
    pthread_cond_destroy(&cond_var);
	
	// measure the end time here
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	// print out the execution time here
	printf("Execution time = %f sec\n", time);		
	
	if (!(fp=fopen(output_file,"wb"))) {
		printf("can not opern file\n");
		return 1;
	}	
	fwrite(a, sizeof(unsigned char),w*h, fp);
    fclose(fp);
    
    return 0;
}