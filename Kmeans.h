class KMeans{
private:
	int dimension_data;
	int number_cluster;
public:
	double **centroid;

	KMeans(int dimension_data, int number_cluster);
	~KMeans();

	// forgy method
	void Initialize(int number_data, double **data);

	int Classify(double data[]);

	double Cluster(int number_data, double **data);
};