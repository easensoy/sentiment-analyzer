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
        return self.predict_with_scores(X)[0]

    def predict_with_scores(self, X):
        """Labels and signed decision scores from a single decision_function call.

        Application policy: a score above 0 is positive; a score of 0 or below is negative.
        SVC.predict follows libsvm's own convention for an exact tie, which need not match
        this policy, so the label is derived from the score here and always agrees with it.
        """
        scores = self.decision_function(X)
        return self.model.classes_[(scores > 0).astype(int)], scores

    def decision_function(self, X):
        """The linear SVM's signed decision score for each row: above 0 leans positive. Not a probability."""
        return self.model.decision_function(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)
