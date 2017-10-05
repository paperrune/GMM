class Matrix{
private:
	int number_row;
	int number_column;
	
	int **index_M;
	int **index_N;
public:
	Matrix();
	~Matrix();
	
	void Inverse(char type_matrix[], int number_row, float **M, float **N);
	void Inverse(char type_matrix[], int number_row, double **M, double **N);
	void Multiplication(int M_row, int M_column, int N_column, float **M, float **N, float **O);
	void Multiplication(int M_row, int M_column, int N_column, double **M, double **N, double **O);
	void Transpose(int number_row, int number_column, float **M, float **N);
	void Transpose(int number_row, int number_column, double **M, double **N);

	float Determinant(char type_matrix[], int number_row, float **M);
	float LU_Decomposition(int number_row, float **M, float **L, float **U);

	double Determinant(char type_matrix[], int number_row, double **M);
	double LU_Decomposition(int number_row, double **M, double **L, double **U);
};