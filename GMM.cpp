#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "GMM.h"
#include "KMeans.h"
#include "Matrix.h"

Gaussian_Mixture_Model::Gaussian_Mixture_Model(char type_covariance[], int dimension_data, int number_gaussian_component){
	strcpy(this->type_covariance, type_covariance);
	this->dimension_data			= dimension_data;
	this->number_gaussian_component = number_gaussian_component;

	covariance	= new double**[number_gaussian_component];
	mean		= new double*[number_gaussian_component];
	weight		= new double[number_gaussian_component];
			
	for(int i = 0;i < number_gaussian_component;i++){
		covariance[i]	= new double*[dimension_data];
		mean[i]			= new double[dimension_data];
			
		for(int j = 0;j < dimension_data;j++){
			covariance[i][j] = new double[dimension_data];
		}
	}
}
Gaussian_Mixture_Model::~Gaussian_Mixture_Model(){
	for(int i = 0;i < number_gaussian_component;i++){
		for(int j = 0;j < dimension_data;j++){
			delete[] covariance[i][j];
		}
		delete[] mean[i];
		delete[] covariance[i];
	}
	delete[] covariance;
	delete[] mean;
	delete[] weight;
}

void Gaussian_Mixture_Model::Initialize(int number_data, double **data){
	KMeans kmeans = KMeans(dimension_data, number_gaussian_component);

	kmeans.Initialize(number_data, data);
	while(kmeans.Cluster(number_data, data));

	for(int i = 0;i < number_gaussian_component;i++){
		for(int j = 0;j < dimension_data;j++){	
			for(int k = 0;k < dimension_data;k++){
				covariance[i][j][k] = (j == k);
			}
		}
		for(int j = 0;j < dimension_data;j++){				
			mean[i][j] = kmeans.centroid[i][j];
		}
		weight[i] = 1.0 / number_gaussian_component;
	}
}
void Gaussian_Mixture_Model::Load_Parameter(char path[]){
	FILE *file = fopen(path, "rt");

	if(file){
		for(int i = 0;i < number_gaussian_component;i++){
			fscanf(file, "%lf", &weight[i]);
		}

		for(int i = 0;i < number_gaussian_component;i++){
			for(int j = 0;j < dimension_data;j++){
				fscanf(file, "%lf", &mean[i][j]);
			}
		}

		for(int i = 0;i < number_gaussian_component;i++){
			for(int j = 0;j < dimension_data;j++){
				for(int k = 0;k < dimension_data;k++){
					fscanf(file, "%lf", &covariance[i][j][k]);
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Load_Parameter], %s not found\n", path);
	}
}
void Gaussian_Mixture_Model::Save_Parameter(char path[]){
	FILE *file = fopen(path, "wt");

	for(int i = 0;i < number_gaussian_component;i++){
		fprintf(file, "%f\n", weight[i]);
	}

	for(int i = 0;i < number_gaussian_component;i++){
		for(int j = 0;j < dimension_data;j++){
			fprintf(file, "%f\n", mean[i][j]);
		}
	}

	for(int i = 0;i < number_gaussian_component;i++){
		for(int j = 0;j < dimension_data;j++){
			for(int k = 0;k < dimension_data;k++){
				fprintf(file, "%f\n", covariance[i][j][k]);
			}
		}
	}
	fclose(file);
}

int Gaussian_Mixture_Model::Classify(double data[]){
	int argmax = -1;

	double max = 0;
		
	for(int i = 0;i < number_gaussian_component;i++){
		double likelihood = weight[i] * Gaussian_Distribution(data, i);

		if(max < likelihood){
			argmax = i;
			max = likelihood;
		}
	}
	return argmax;
}

double Gaussian_Mixture_Model::Calculate_Likelihood(double data[]){
	double likelihood = 0;
		
	for(int i = 0;i < number_gaussian_component;i++){
		likelihood += weight[i] * Gaussian_Distribution(data, i);
	}
	return likelihood;
}
double Gaussian_Mixture_Model::Calculate_Likelihood(double data[], double gaussian_distribution[]){
	double likelihood = 0;
		
	for(int i = 0;i < number_gaussian_component;i++){
		likelihood += weight[i] * gaussian_distribution[i];
	}
	return likelihood;
}
double Gaussian_Mixture_Model::Expectaion_Maximization(int number_data, double **data){
	double log_likelihood = 0;

	double *gaussian_distribution	= new double[number_gaussian_component];
	double *sum_likelihood			= new double[number_gaussian_component];
		
	double **new_mean = new double*[number_gaussian_component];
		
	double ***new_covariance = new double**[number_gaussian_component];
		
	for(int i = 0;i < number_gaussian_component;i++){
		new_mean[i]			= new double[dimension_data];
		new_covariance[i]	= new double*[dimension_data];
			
		for(int j = 0;j < dimension_data;j++){
			new_covariance[i][j] = new double[dimension_data];
		}
	}

	for(int i = 0;i < number_gaussian_component;i++){
		for(int j = 0;j < dimension_data;j++){
			for(int k = 0;k < dimension_data;k++){
				new_covariance[i][j][k] = 0;
			}
			new_mean[i][j] = 0;
		}
		sum_likelihood[i] = 0;
	}

	for(int i = 0;i < number_data;i++){
		double sum = 0;
				
		for(int j = 0;j < number_gaussian_component;j++){
			sum += weight[j] * (gaussian_distribution[j] = Gaussian_Distribution(data[i], mean[j], covariance[j]));
		}
		for(int j = 0;j < number_gaussian_component;j++){
			double likelihood = weight[j] * gaussian_distribution[j] / sum;

			for(int k = 0;k < dimension_data;k++){
				for(int l = 0;l < dimension_data;l++){
					new_covariance[j][k][l] += likelihood * (data[i][k] - mean[j][k]) * (data[i][l] - mean[j][l]);
				}
				new_mean[j][k] += likelihood * data[i][k];
			}
			sum_likelihood[j] += likelihood;
		}
	}

	for(int i = 0;i < number_gaussian_component;i++){						
		for(int j = 0;j < dimension_data;j++){
			for(int k = 0;k < dimension_data;k++){
				covariance[i][j][k] = new_covariance[i][j][k] / sum_likelihood[i];
			}
			mean[i][j] = new_mean[i][j] / sum_likelihood[i];
		}
		weight[i] = sum_likelihood[i] / number_data;
	}
			
	for(int i = 0;i < number_data;i++){
		log_likelihood += log(Calculate_Likelihood(data[i]));
	}
		
	for(int i = 0;i < number_gaussian_component;i++){
		for(int j = 0;j < dimension_data;j++){
			delete[] new_covariance[i][j];
		}
		delete[] new_mean[i];
		delete[] new_covariance[i];
	}
	delete[] gaussian_distribution;
	delete[] new_covariance;
	delete[] new_mean;
	delete[] sum_likelihood;

	return log_likelihood;
}
double Gaussian_Mixture_Model::Gaussian_Distribution(double data[], int component_index){
	return Gaussian_Distribution(data, mean[component_index], covariance[component_index]);
}
double Gaussian_Mixture_Model::Gaussian_Distribution(double data[], double mean[], double **covariance){
	double result;
	double sum = 0;
		
	double *partial_sum = new double[dimension_data];
		
	double **inversed_covariance = new double*[dimension_data];

	Matrix matrix;
		
	for(int i = 0;i < dimension_data;i++){
		inversed_covariance[i] = new double[dimension_data];
	}
	matrix.Inverse(type_covariance, dimension_data, covariance, inversed_covariance);
		
	for(int i = 0;i < dimension_data;i++){
		double sum = 0;
			
		for(int j = 0;j < dimension_data;j++){
			sum += (data[j] - mean[j]) * inversed_covariance[j][i];
		}
		partial_sum[i] = sum;
	}
	for(int i = 0;i < dimension_data;i++){
		sum += partial_sum[i] * (data[i] - mean[i]);
	}
		
	for(int i = 0;i < dimension_data;i++){
		delete[] inversed_covariance[i];
	}
	delete[] inversed_covariance;
	delete[] partial_sum;
		
	result = 1.0 / (pow(2 * 3.1415926535897931, dimension_data / 2.0) * sqrt(matrix.Determinant(type_covariance, dimension_data, covariance))) * exp(-0.5 * sum);
		
	if(_isnan(result) || !_finite(result)){
		fprintf(stderr, "[Gaussian Distribution], [The covariance matrix is rank deficient], [result: %lf]\n", result);
	}
	return result;
}