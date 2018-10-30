from params import *

class LeastSquares():
    def __init__(self):
        self.theta = None

    def fit(self, phi, y):
        self.theta = np.linalg.inv(phi.dot(phi.T)).dot(phi).dot(y)

    def predict(self, phi):
        return phi.T.dot(self.theta)

    def save(self, model_path):
        np.save(model_path, self.theta)

    def load(self, model_path):
        self.theta = np.load(model_path)