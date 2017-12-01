from sklearn.preprocessing import StandardScaler
from sklearn_utils.utils import filter_by_label


class StandardScalerByLabel(StandardScaler):
    """StandardScaler for using only by give label."""

    def __init__(self, reference_label):
        """
        :reference_label: the label scaling will be performed by.
        """
        super().__init__()
        self.reference_label = reference_label

    def partial_fit(self, X, y):
        """
        :X: {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        :y: Healthy 'h' or 'sick_name'
        """
        X, y = filter_by_label(X, y, self.reference_label)
        super().partial_fit(X, y)
        return self
