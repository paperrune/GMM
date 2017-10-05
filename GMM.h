class Gaussian_Mixture_Model{
private:
	char type_covariance[16];

	int dimension_data;
	int number_gaussian_component;
public:
	double *weight;
	
	double **mean;

	double ***covariance;

	Gaussian_Mixture_Model(char type_covariance[], int dimension_data, int number_gaussian_component);
	~Gaussian_Mixture_Model();

	void Initialize(int number_data, double **data);
	void Load_Parameter(char path[]);
	void Save_Parameter(char path[]);

	int Classify(double data[]);

	double Calculate_Likelihood(double data[]);
	double Calculate_Likelihood(double data[], double gaussian_distribution[]);
	double Expectaion_Maximization(int number_data, double **data);
	double Gaussian_Distribution(double data[], int component_index);
	double Gaussian_Distribution(double data[], double mean[], double **covariance);
};