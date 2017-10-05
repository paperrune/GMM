#include <stdio.h>
#include <stdlib.h>

#include "GMM.h"

int main(){
	int dimension_data	 = 2;
	int number_data		 = 300;
	int number_iteration = 100;

	int number_gaussian_component = 4;

	double **data = new double*[number_data];

	FILE *file;

	Gaussian_Mixture_Model GMM = Gaussian_Mixture_Model("full", dimension_data, number_gaussian_component);

	for(int i = 0;i < number_data;i++){
		double position[] = {0.25, 0.75};

		data[i]		= new double[dimension_data];
		data[i][0]	= 0.25 * rand() / RAND_MAX - 0.125 + position[(i % 2)];
		data[i][1]	= 0.25 * rand() / RAND_MAX - 0.125 + position[(i % 4) / 2];
	}

	printf("step	log_likelihood\n");
	for(int i = 0;i < number_iteration;i++){
		double log_likelihood;

		if(i == 0) GMM.Initialize(number_data, data);

		log_likelihood = GMM.Expectaion_Maximization(number_data, data);
		printf("%d	%lf\n", i + 1, log_likelihood);
	}

	printf("\nmean\n");
	for(int i = 0;i < number_gaussian_component;i++){
		for(int j = 0;j < dimension_data;j++){
			printf("%lf ", GMM.mean[i][j]);
		}
		printf("\n");
	}

	file = fopen("result.txt", "wt");

	for(int j = 0;j < number_gaussian_component;j++){
		for(int i = 0;i < number_data;i++){
			if(GMM.Classify(data[i]) == j){
				fprintf(file, "%d %lf %lf\n", GMM.Classify(data[i]), data[i][0], data[i][1]);
			}
		}
	}
	fclose(file);

	for(int i = 0;i < number_data;i++){
		delete[] data[i];
	}
	delete[] data;

	return 0;
}