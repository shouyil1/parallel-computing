#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define h  800 
#define w  800

#define input_file  "input.raw"
#define output_file "output.raw"

int main(int argc, char** argv){
    int i, j, k;
    FILE *fp;
	struct timespec start, stop; 
	double time;

  	unsigned char *a = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
    
	// the matrix is stored in a linear array in row major fashion
	if (!(fp=fopen(input_file, "rb"))) {
		printf("can not opern file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), w*h, fp);
	fclose(fp);
    
	// measure the start time here
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

	//  Your code goes here
	int cluster[6] = {0, 65, 100, 125, 190, 255};  // initial mean values
    int sum[6], count[6], min_distance, distance, cluster_index;

    for (k = 0; k < 50; k++) {  // run 50 iterations
        // Initialize sum and count for each cluster
        for (i = 0; i < 6; i++) {
            sum[i] = 0;
            count[i] = 0;
        }

        // Assign each data element to the closest cluster and update sum and count
        for (i = 0; i < h*w; i++) {
            min_distance = 255;
            cluster_index = 0;
			// finding the closest cluster
            for (j = 0; j < 6; j++) {
                distance = abs(a[i] - cluster[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    cluster_index = j;
                }
            }
            sum[cluster_index] += a[i];
            count[cluster_index]++;
        }

        // Recompute the mean value of each cluster
        for (i = 0; i < 6; i++) {
            if (count[i] != 0) {
                cluster[i] = sum[i] / count[i];
            }
        }
    }

    // Replace the value of each data with the mean value of the cluster it belongs to
    for (i = 0; i < h*w; i++) {
        min_distance = 255;
        cluster_index = 0;
        for (j = 0; j < 6; j++) {
            distance = abs(a[i] - cluster[j]);
            if (distance < min_distance) {
                min_distance = distance;
                cluster_index = j;
            }
        }
        a[i] = cluster[cluster_index];
    }
	
		
	
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