from sklearn.svm import SVC
import joblib

class SentimentModel:
    def __init__(self):
        self.model = SVC(kernel='linear')
        self.preprocessor = None

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)
