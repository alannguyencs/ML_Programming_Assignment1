from params import *

class BayesianRegression():
    def __init__(self):
        self.mean = None
        self.variance = None

    def fit(self, phi, y, _alpha):
        d = phi.shape[0]
        I_d = 1./_alpha * np.identity(d)
        self.variance = np.linalg.inv(I_d + phi.dot(phi.T) * (1./(_alpha**2)))
        self.mean = 1./(_alpha**2) * self.variance.dot(phi).dot(y)

    def predict(self, phi):
        mean_star = (phi.T).dot(self.mean)
        variance_star = ((phi.T).dot(self.variance).dot(phi)).diagonal()
        variance_star = variance_star[:, np.newaxis]
        std_star = np.asarray([math.sqrt(variance_star[i]) for i in range(variance_star.shape[0])])
        return mean_star, std_star

    def save(self, model_path):
        np.save(model_path, self.theta)

    def load(self, model_path):
        self.theta = np.load(model_path)