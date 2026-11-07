from sklearn.svm import SVC
import joblib

LABELS = {0: "negative", 1: "positive"}


class SentimentModel:
    def __init__(self):
        self.model = SVC(kernel='linear')
        self.preprocessor = None

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        # The margin's sign decides, so a label can never contradict its margin; libsvm's own
        # predict sends an exact tie (margin 0) the other way. A tie counts as negative here.
        return self.model.classes_[(self.decision_function(X) > 0).astype(int)]

    def decision_function(self, X):
        """Signed distance from the SVM's separating hyperplane: above 0 means positive. Not a probability."""
        return self.model.decision_function(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)
