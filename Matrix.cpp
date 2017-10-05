#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "Matrix.h"

Matrix::Matrix(){
	number_row		= 0;
	number_column	= 0;
		
	index_M = new int*[0];
	index_N = new int*[0];
}
Matrix::~Matrix(){
	for(int i = 0;i < number_column;i++){
		delete[] index_N[i];
	}
	for(int i = 0;i < number_row;i++){
		delete[] index_M[i];
	}
	delete[] index_M;
	delete[] index_N;
}

void Matrix::Inverse(char type_matrix[], int number_row, float **M, float **N){
	int m = number_row;

	if(m == 1){
		N[0][0] = 1 / M[0][0];
	}
	else
	if(m >= 2){
		if(strcmp(type_matrix, "diagonal") == 0){
			for(int i = 0;i < m;i++){
				for(int j = 0;j < m;j++){
					N[i][j] = (i == j) ? (1 / M[i][j]):(0);
				}
			}
		}
		else
		if(strcmp(type_matrix, "block-diagonal") == 0){				
			int recent_index = 0;
				
			for(int i = 0;i < m;i++){
				for(int j = 0;j < m;j++){
					N[i][j] = 0;
				}
			}
			for(int i = 0;i < m;i++){
				if(M[recent_index][i] == 0 || i == m - 1){
					float **T;
						
					if(i == m - 1){
						i = m;
					}
						
					T = new float*[i - recent_index];
				
					for(int j = 0;j < i - recent_index;j++){
						T[j] = new float[i - recent_index];
					}
						
					for(int j = 0;j < i - recent_index;j++){							
						for(int k = 0;k < i - recent_index;k++){
							T[j][k] = M[recent_index + j][recent_index + k];
						}
					}
					Inverse("full", i - recent_index, T, T);
						
					for(int j = 0;j < i - recent_index;j++){							
						for(int k = 0;k < i - recent_index;k++){
							N[recent_index + j][recent_index + k] = T[j][k];
						}
					}
						
					for(int j = 0;j < i - recent_index;j++){
						delete[] T[j];
					}
					delete[] T;
						
					recent_index = i;
				}
			}
		}
		else
		if(strcmp(type_matrix, "full") == 0){
			float **T = new float*[m];
				
			for(int i = 0;i < m;i++){
				T[i] = new float[m];
			}
				
			for(int i = 0;i < m;i++){			
				for(int j = 0;j < m;j++){
					T[i][j] = M[i][j];
				}
			}
				
			// make identity matrix
			for(int i = 0;i < m;i++){
			    for(int j = 0;j < m;j++){
			        N[i][j] = (i == j) ? (1):(0);
			    }
			}
			    
			// lower triangle elimination
			for(int k = 0;k < m - 1;k++){
			    for(int i = k + 1;i < m;i++){
			        float ratio = T[i][k] / T[k][k];
			        	
			        for(int j = k;j < m;j++){
			            T[i][j] -= T[k][j] * ratio;
			        }
			        for(int j = 0;j < m;j++){
			            N[i][j] -= N[k][j] * ratio;
			        }
			    }
			}
			 
			// make diagonal to 1.0 
			for(int i = 0;i < m;i++){
			    float ratio = T[i][i];
			        
			    T[i][i] = 1.0;
			    for(int j = i + 1;j < m;j++){
			        T[i][j] /= ratio;
			    }
			    for(int j = 0;j <= i;j++){
			        N[i][j] /= ratio;
			    }
			}
			    
			// upper triangle elimination
			for(int k = m - 1;k > 0;k--){
			    for(int i = k - 1;i >= 0;i--){
			        float ratio = T[i][k];
			            
			        T[i][k] = 0;
			        for(int j = 0;j < m;j++){
			            N[i][j] -= N[k][j] * ratio;
			        }
			    }
			}
				
			for(int i = 0;i < m;i++){
				delete[] T[i];
			}
			delete[] T;
		}
	}
}
void Matrix::Inverse(char type_matrix[], int number_row, double **M, double **N){
	int m = number_row;

	if(m == 1){
		N[0][0] = 1 / M[0][0];
	}
	else
	if(m >= 2){
		if(strcmp(type_matrix, "diagonal") == 0){
			for(int i = 0;i < m;i++){
				for(int j = 0;j < m;j++){
					N[i][j] = (i == j) ? (1 / M[i][j]):(0);
				}
			}
		}
		else
		if(strcmp(type_matrix, "block-diagonal") == 0){				
			int recent_index = 0;
				
			for(int i = 0;i < m;i++){
				for(int j = 0;j < m;j++){
					N[i][j] = 0;
				}
			}
			for(int i = 0;i < m;i++){
				if(M[recent_index][i] == 0 || i == m - 1){
					double **T;
						
					if(i == m - 1){
						i = m;
					}
						
					T = new double*[i - recent_index];
				
					for(int j = 0;j < i - recent_index;j++){
						T[j] = new double[i - recent_index];
					}
						
					for(int j = 0;j < i - recent_index;j++){							
						for(int k = 0;k < i - recent_index;k++){
							T[j][k] = M[recent_index + j][recent_index + k];
						}
					}
					Inverse("full", i - recent_index, T, T);
						
					for(int j = 0;j < i - recent_index;j++){							
						for(int k = 0;k < i - recent_index;k++){
							N[recent_index + j][recent_index + k] = T[j][k];
						}
					}
						
					for(int j = 0;j < i - recent_index;j++){
						delete[] T[j];
					}
					delete[] T;
						
					recent_index = i;
				}
			}
		}
		else
		if(strcmp(type_matrix, "full") == 0){
			double **T = new double*[m];
				
			for(int i = 0;i < m;i++){
				T[i] = new double[m];
			}
				
			for(int i = 0;i < m;i++){			
				for(int j = 0;j < m;j++){
					T[i][j] = M[i][j];
				}
			}
				
			// make identity matrix
			for(int i = 0;i < m;i++){
			    for(int j = 0;j < m;j++){
			        N[i][j] = (i == j) ? (1):(0);
			    }
			}
			    
			// lower triangle elimination
			for(int k = 0;k < m - 1;k++){
			    for(int i = k + 1;i < m;i++){
			        double ratio = T[i][k] / T[k][k];
			        	
			        for(int j = k;j < m;j++){
			            T[i][j] -= T[k][j] * ratio;
			        }
			        for(int j = 0;j < m;j++){
			            N[i][j] -= N[k][j] * ratio;
			        }
			    }
			}
			 
			// make diagonal to 1.0 
			for(int i = 0;i < m;i++){
			    double ratio = T[i][i];
			        
			    T[i][i] = 1.0;
			    for(int j = i + 1;j < m;j++){
			        T[i][j] /= ratio;
			    }
			    for(int j = 0;j <= i;j++){
			        N[i][j] /= ratio;
			    }
			}
			    
			// upper triangle elimination
			for(int k = m - 1;k > 0;k--){
			    for(int i = k - 1;i >= 0;i--){
			        double ratio = T[i][k];
			            
			        T[i][k] = 0;
			        for(int j = 0;j < m;j++){
			            N[i][j] -= N[k][j] * ratio;
			        }
			    }
			}
				
			for(int i = 0;i < m;i++){
				delete[] T[i];
			}
			delete[] T;
		}
	}
}
void Matrix::Multiplication(int M_row, int M_column, int N_column, float **M, float **N, float **O){
	int m = M_row;
	int n = N_column;
	int o = M_column;
		
	int **index_M = new int*[m];
	int **index_N = new int*[n];
		
	float **T = new float*[m];
		
	for(int i = 0;i < m;i++){
		index_M[i]	= new int[2];
		T[i]		= new float[n];
	}
	for(int i = 0;i < n;i++){
		index_N[i] = new int[2];
	}
		
	for(int i = 0;i < m;i++){			
		for(int j = 0;j < o;j++){
			if(M[i][j] != 0){
				index_M[i][0] = j;
				break;
			}
		}
		for(int j = o - 1;j >= 0;j--){
			if(M[i][j] != 0){
				index_M[i][1] = j + 1;
				break;
			}
		}
	}
	for(int i = 0;i < n;i++){				
		for(int j = 0;j < o;j++){
			if(N[j][i] != 0){
				index_N[i][0] = j;
				break;
			}
		}
		for(int j = o - 1;j >= 0;j--){
			if(N[j][i] != 0){
				index_N[i][1] = j + 1;
				break;
			}
		}
	}		
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			int index = (index_M[i][1] < index_N[j][1]) ? (index_M[i][1]):(index_N[j][1]);
				
			float sum = 0;
	
			for(int k = (index_M[i][0] > index_N[j][0]) ? (index_M[i][0]):(index_N[j][0]);k < index;k++){
				sum += M[i][k] * N[k][j];
			}
			T[i][j] = sum;
		}
	}			
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			O[i][j] = T[i][j];
		}
	}
		
	for(int i = 0;i < m;i++){
		delete[] index_M[i];
		delete[] T[i];
	}
	for(int i = 0;i < n;i++){
		delete[] index_N[i];
	}		
	delete[] index_M;
	delete[] index_N;
	delete[] T;
}
void Matrix::Multiplication(int M_row, int M_column, int N_column, double **M, double **N, double **O){
	int m = M_row;
	int n = N_column;
	int o = M_column;
		
	int **index_M = new int*[m];
	int **index_N = new int*[n];
		
	double **T = new double*[m];
		
	for(int i = 0;i < m;i++){
		index_M[i]	= new int[2];
		T[i]		= new double[n];
	}
	for(int i = 0;i < n;i++){
		index_N[i] = new int[2];
	}
		
	for(int i = 0;i < m;i++){			
		for(int j = 0;j < o;j++){
			if(M[i][j] != 0){
				index_M[i][0] = j;
				break;
			}
		}
		for(int j = o - 1;j >= 0;j--){
			if(M[i][j] != 0){
				index_M[i][1] = j + 1;
				break;
			}
		}
	}
	for(int i = 0;i < n;i++){				
		for(int j = 0;j < o;j++){
			if(N[j][i] != 0){
				index_N[i][0] = j;
				break;
			}
		}
		for(int j = o - 1;j >= 0;j--){
			if(N[j][i] != 0){
				index_N[i][1] = j + 1;
				break;
			}
		}
	}		
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			int index = (index_M[i][1] < index_N[j][1]) ? (index_M[i][1]):(index_N[j][1]);
				
			double sum = 0;
	
			for(int k = (index_M[i][0] > index_N[j][0]) ? (index_M[i][0]):(index_N[j][0]);k < index;k++){
				sum += M[i][k] * N[k][j];
			}
			T[i][j] = sum;
		}
	}			
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			O[i][j] = T[i][j];
		}
	}
		
	for(int i = 0;i < m;i++){
		delete[] index_M[i];
		delete[] T[i];
	}
	for(int i = 0;i < n;i++){
		delete[] index_N[i];
	}		
	delete[] index_M;
	delete[] index_N;
	delete[] T;
}
void Matrix::Transpose(int number_row, int number_column, float **M, float **N){
	int m = number_row;
	int n = number_column;
		
	float **T = new float*[m];
		
	for(int i = 0;i < m;i++){
		T[i] = new float[n];
	}
		
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			T[i][j] = M[j][i];
		}
	}
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			N[i][j] = T[i][j];
		}
	}
		
	for(int i = 0;i < m;i++){
		delete[] T[i];
	}
	delete[] T;
}
void Matrix::Transpose(int number_row, int number_column, double **M, double **N){
	int m = number_row;
	int n = number_column;
		
	double **T = new double*[m];
		
	for(int i = 0;i < m;i++){
		T[i] = new double[n];
	}
		
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			T[i][j] = M[j][i];
		}
	}
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			N[i][j] = T[i][j];
		}
	}
		
	for(int i = 0;i < m;i++){
		delete[] T[i];
	}
	delete[] T;
}

float Matrix::Determinant(char type_matrix[], int number_row, float **M){
	int m = number_row;
		
	float determinant = 1;
		
	if(m == 1){
		determinant = M[0][0];
	}
	else
	if(m == 2){
		determinant = M[0][0] * M[1][1] - M[0][1] * M[1][0];
	}
	else
	if(m >= 3){
		if(!strcmp(type_matrix, "diagonal")){
			for(int i = 0;i < m;i++){
				determinant *= M[i][i];
			}
		}
		else
		if(!strcmp(type_matrix, "block-diagonal")){
			int recent_index = 0;
				
			for(int i = 0;i < m;i++){
				if(M[recent_index][i] == 0 || i == m - 1){
					float **T;
						
					if(i == m - 1){
						i = m;
					}
						
					T = new float*[i - recent_index];
						
					for(int j = 0;j < i - recent_index;j++){
						T[j] = new float[i - recent_index];
					}						
					for(int j = 0;j < i - recent_index;j++){							
						for(int k = 0;k < i - recent_index;k++){
							T[j][k] = M[recent_index + j][recent_index + k];
						}
					}
					determinant *= Determinant("full", i - recent_index, T);
						
					for(int j = 0;j < i - recent_index;j++){
						delete[] T[j];
					}
					delete[] T;
						
					recent_index = i;
				}
			}
		}
		else
		if(!strcmp(type_matrix, "full")){
			float **L = new float*[m];
			float **U = new float*[m];
				
			for(int i = 0;i < m;i++){
				L[i] = new float[m];
				U[i] = new float[m];
			}
			if(LU_Decomposition(m, M, L, U) == 0){
				determinant = 0;
			}			
			for(int i = 0;i < m;i++){
				determinant *= U[i][i];
					
				delete[] L[i];
				delete[] U[i];
			}
			delete[] L;
			delete[] U;
		}
	}		
	return determinant;
}
float Matrix::LU_Decomposition(int number_row, float **M, float **L, float **U){
	int m = number_row;
		
	for(int i = 0;i < m;i++){
		L[i][i] = 1;
		 
		for(int j = i;j < m;j++){
		    float sum = 0;
		        
		    for(int k = 0;k <= i - 1;k++){
		        sum += L[i][k] * U[k][j];
		    }
		    U[i][j] = M[i][j] - sum;
		}
		for(int j = i + 1;j < m;j++){
		    float sum = 0;

		    for(int k = 0;k <= i - 1;k++){
		        sum += L[j][k] * U[k][i];
		    }
			if(U[i][i] == 0){
				return 0;
			}
			L[j][i] = (M[j][i] - sum) / U[i][i];
		}
	}
}

double Matrix::Determinant(char type_matrix[], int number_row, double **M){
	int m = number_row;
		
	double determinant = 1;
		
	if(m == 1){
		determinant = M[0][0];
	}
	else
	if(m == 2){
		determinant = M[0][0] * M[1][1] - M[0][1] * M[1][0];
	}
	else
	if(m >= 3){
		if(!strcmp(type_matrix, "diagonal")){
			for(int i = 0;i < m;i++){
				determinant *= M[i][i];
			}
		}
		else
		if(!strcmp(type_matrix, "block-diagonal")){
			int recent_index = 0;
				
			for(int i = 0;i < m;i++){
				if(M[recent_index][i] == 0 || i == m - 1){
					double **T;
						
					if(i == m - 1){
						i = m;
					}
						
					T = new double*[i - recent_index];
						
					for(int j = 0;j < i - recent_index;j++){
						T[j] = new double[i - recent_index];
					}						
					for(int j = 0;j < i - recent_index;j++){							
						for(int k = 0;k < i - recent_index;k++){
							T[j][k] = M[recent_index + j][recent_index + k];
						}
					}
					determinant *= Determinant("full", i - recent_index, T);
						
					for(int j = 0;j < i - recent_index;j++){
						delete[] T[j];
					}
					delete[] T;
						
					recent_index = i;
				}
			}
		}
		else
		if(!strcmp(type_matrix, "full")){
			double **L = new double*[m];
			double **U = new double*[m];
				
			for(int i = 0;i < m;i++){
				L[i] = new double[m];
				U[i] = new double[m];
			}
			if(LU_Decomposition(m, M, L, U) == 0){
				determinant = 0;
			}			
			for(int i = 0;i < m;i++){
				determinant *= U[i][i];
					
				delete[] L[i];
				delete[] U[i];
			}
			delete[] L;
			delete[] U;
		}
	}		
	return determinant;
}
double Matrix::LU_Decomposition(int number_row, double **M, double **L, double **U){
	int m = number_row;
		
	for(int i = 0;i < m;i++){
		L[i][i] = 1;
		 
		for(int j = i;j < m;j++){
		    double sum = 0;
		        
		    for(int k = 0;k <= i - 1;k++){
		        sum += L[i][k] * U[k][j];
		    }
		    U[i][j] = M[i][j] - sum;
		}
		for(int j = i + 1;j < m;j++){
		    double sum = 0;

		    for(int k = 0;k <= i - 1;k++){
		        sum += L[j][k] * U[k][i];
		    }
			if(U[i][i] == 0){
				return 0;
			}
			L[j][i] = (M[j][i] - sum) / U[i][i];
		}
	}
}