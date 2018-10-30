from params import *
from cvxopt import matrix, solvers

class Lasso():
    def __init__(self):
        self.theta = None

    def fit(self, phi, y, _lambda):
        dimension = phi.shape[0]
        phi_phi_T = phi.dot(phi.T)
        H = np.hstack((np.vstack((phi_phi_T, -phi_phi_T)), np.vstack((phi_phi_T, phi_phi_T))))
        phi_Y = phi.dot(y)
        f = _lambda * np.ones((2 * dimension, 1)) - np.vstack((phi_Y, -phi_Y))
        G = - np.identity(2 * dimension)
        h = np.zeros(2 * dimension)

        H = matrix(H)
        f = matrix(f)
        G = matrix(G)
        h = matrix(h)

        res = solvers.qp(H, f, G, h, options={'show_progress': False})
        res = np.array(res['x'])
        self.theta = res[:dimension] - res[dimension:]

    def predict(self, phi):
        return phi.T.dot(self.theta)

    def save(self, model_path):
        np.save(model_path, self.theta)

    def load(self, model_path):
        self.theta = np.load(model_path)