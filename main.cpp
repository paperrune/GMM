#include <fstream>

#include "GMM.h"

int main(){
	string type_covariance = "full"; // <-> "diagonal"

	int dimension_data = 2;
	int number_data = 100;
	int number_iterations = 200;

	int number_gaussian_components = 4;

	double **data = new double*[number_data];

	ofstream file;

	Gaussian_Mixture_Model GMM = Gaussian_Mixture_Model(type_covariance, dimension_data, number_gaussian_components);

	for (int i = 0; i < number_data; i++){
		double position[] = { 0.25, 0.75 };

		data[i] = new double[dimension_data];
		data[i][0] = 0.25 * rand() / RAND_MAX - 0.125 + position[(i % 2)];
		data[i][1] = 0.25 * rand() / RAND_MAX - 0.125 + position[(i % 4) / 2];
	}

	printf("step	log_likelihood\n");
	for (int i = 0; i < number_iterations; i++){
		double log_likelihood;

		if (i == 0) GMM.Initialize(number_data, data);

		log_likelihood = GMM.Expectaion_Maximization(number_data, data);
		if ((i + 1) % 10 == 0) printf("%d	%lf\n", i + 1, log_likelihood);
	}

	printf("\nmean\n");
	for (int i = 0; i < number_gaussian_components; i++){
		for (int j = 0; j < dimension_data; j++){
			printf("%lf ", GMM.mean[i][j]);
		}
		printf("\n");
	}

	file.open("result.txt");

	for (int j = 0; j < number_gaussian_components; j++){
		for (int i = 0; i < number_data; i++){
			if (GMM.Classify(data[i]) == j){
				file << GMM.Classify(data[i]) << " " << data[i][0] << " " << data[i][1] << endl;
			}
		}
	}
	file.close();

	for (int i = 0; i < number_data; i++){
		delete[] data[i];
	}
	delete[] data;

	return 0;
}
