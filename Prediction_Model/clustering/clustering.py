import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from Prediction_Model.file_operations.file_methods import FileOperations
from Prediction_Model.config.config import PACAKAGE_ROOT, RANDOM_SEED

class KMeansClustering:
    def __init__(self, logger_file, logger):
        """
        Initialize the KMeansClustering object with logger file and logger instance.

        Parameters:
        logger_file (str): File path for logging.
        logger: Logger instance for logging.
        """
        self.log_file = logger_file
        self.logger = logger

    def elbow_plot(self, data, max_clusters=10):
        """
        Find the optimal number of clusters for KMeans clustering using the elbow method.

        Parameters:
        data (numpy.ndarray or pandas.DataFrame): The dataset to be clustered.
        max_clusters (int): The maximum number of clusters to consider. Defaults to 10.

        Returns:
        int: The optimal number of clusters.
        """
        self.logger.add_log(self.log_file, 'Entered the elbow_plot method of the KMeansClustering class')
        distortions = []
        try:
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_SEED)
                kmeans.fit(data)
                distortions.append(kmeans.inertia_)

            # Plot the elbow plot
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, max_clusters + 1), distortions, marker='o', linestyle='--')
            plt.xlabel('Number of clusters')
            plt.ylabel('Distortion')
            plt.title('Elbow Plot')
            plt.xticks(range(1, max_clusters + 1))
            plt.grid(True)
            plt.savefig(f'{PACAKAGE_ROOT}/clustering/K-Means_Elbow.PNG')
            self.logger.add_log(self.log_file, 'Created Elbow plot and saved it.')

            # Find the optimal number of clusters using the KneeLocator
            kn = KneeLocator(range(1, max_clusters + 1), distortions, curve='convex', direction='decreasing')
            optimal_clusters = kn.elbow
            self.logger.add_log(self.log_file, f'Found optimal number of clusters which is {optimal_clusters}')
            return optimal_clusters
        except Exception as e:
            self.logger.add_log(self.log_file, 'Error while creating the elbow plot and finding optimal number of clusters::'+str(e))
            raise Exception()

    def create_clusters(self, dataframe, optimal_clusters):
        """
        Perform KMeans clustering on the provided DataFrame.

        Parameters:
        dataframe (pandas.DataFrame): DataFrame containing data to be clustered.
        optimal_clusters (int): The optimal number of clusters.

        Returns:
        pandas.DataFrame: DataFrame with an additional 'cluster' column indicating cluster assignments.
        """
        self.logger.add_log(self.log_file, 'Entered the create_clusters method of the KMeansClustering class')
        file_operations = FileOperations(self.log_file, self.logger)
        try:
            data = dataframe.copy()
            kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=RANDOM_SEED)
            y_pred = kmeans.fit_predict(data)
            # Save the model
            file_operations.save_model(kmeans, "Kmeans")

            data['cluster'] = y_pred
            self.logger.add_log(self.log_file, 'Trained clustering model and saved it.')
            return data

        except Exception as e:
            self.logger.add_log(self.log_file, 'Error while training clustering model::'+str(e))
            raise Exception()
