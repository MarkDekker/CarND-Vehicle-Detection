"""Classifier designed to identify cars in images."""

import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

class Classifier():
    """Base class for the ML models to identify cars."""
    def __init__(self, classifier):
        self.clf = classifier
        self.feature_scaler = None

    def __call__(self):
        raise NotImplementedError

    def train(self, features_train, labels_train, features_test,
              labels_test):
        """Trains the classifier with the supplied training data."""
        features_train = self.normalise(features_train)
        features_test = self.normalise(features_test)
        self.clf.fit(features_train, labels_train)

        start = time.time()
        test_set_accuracy = self.clf.score(features_test, labels_test)
        self.predict(features_test)
        end = time.time()

        print('Classifier test accuracy = ',
              round(test_set_accuracy, 4),
              '(took', round(end - start, 2), 'seconds on', len(features_test),
              'entries)')

    def predict(self, features):
        """Predicts the labels for a given list of input features."""
        features = self.normalise(features)
        return self.clf.predict(features)

    def normalise(self, features):
        """Normalises the input feature set. """
        if self.feature_scaler is None:
            raise ValueError('The feature scaler has not yet been fit!')
        else:
            return self.feature_scaler.transform(features)

    def fit_feature_scaler(self, training_features):
        """Fit a feature scaler to the training data to easily normalise inputs
        later.
        """
        self.feature_scaler = StandardScaler().fit(training_features)

    @classmethod
    def svm(cls, params=None):
        """Set up a support vector machine classifier."""
        if params is None:
            params = {'C': 1.0,
                      'kernel': 'rbf',
                      'max_iter': -1}
        return cls(SVC(**params))

    @classmethod
    def svm_linear(cls, params=None):
        """Set up a linear support vector machine classifier."""
        if params is None:
            params = {'C': 1.0,
                      'dual': True}
        return cls(LinearSVC(**params))

