from params import *

class DataSet():
    def __init__(self, x_dir, y_dir, percent_sample=100, percent_outlier=0):
        self.x = np.loadtxt(x_dir)
        self.y = np.loadtxt(y_dir).reshape(-1, 1)
        self.size = self.y.shape[0]

        # if self.x.shape[0] != self.size:
        #     self.x = np.transpose(self.x(1,0))

        if percent_sample < 100:
            num_sample = int(percent_sample / 100 * self.size)
            idx = [i for i in range( self.size)]
            shuffle(idx)
            self.x = np.asarray([self.x[idx[i]] for i in range(num_sample)])
            self.y = np.asarray([self.y[idx[i]] for i in range(num_sample)])

        if percent_outlier > 0:
            num_outlier = int(percent_outlier / 100 * num_sample)
            for i in range(num_outlier):
                outlier_param = randint(3, 10)
                self.y[i] *= outlier_param




#
# data = DataSet(x_dir=x_path, y_dir=y_path)
#
#
# print (data.get_data(percent_sample=100, percent_outlier=20))