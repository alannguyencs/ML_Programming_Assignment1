from params import *

class RegularizedLeastSquares():
    def __init__(self):
        self.theta = None

    def fit(self, phi, y, _lambda):
        d = phi.shape[0]
        lambda_I = _lambda * np.identity(d)
        self.theta = np.linalg.inv(phi.dot(phi.T) + lambda_I).dot(phi).dot(y)

    def predict(self, phi):
        return phi.T.dot(self.theta)

    def save(self, model_path):
        np.save(model_path, self.theta)

    def load(self, model_path):
        self.theta = np.load(model_path)