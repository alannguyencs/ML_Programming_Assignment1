from params import *



def generate_polynomial_features(x, polynomial_degree=5):
    phi = [x ** 0]
    for k in range(1, polynomial_degree + 1):
        phi = np.append(phi, [x ** k], axis=0)
    return phi

def generate_counting_features(x):
    # for i in range(9):
    #     print (np.amax(x[i]), np.amin(x[i]))

    x = x.T
    (num_samples, dimension) = x.shape
    #initialize with bias features
    phi = [[] for _ in range(num_samples)]

    #append normalized features
    for i in range(num_samples):
        for j in range(dimension):
            # phi[i].append(x[i][j])
            r = range_counting_feature[j] / 2
            phi[i].append((x[i][j] - min_counting_feature[j] - r) / r)

    # append 2nd order polinomial features
    for i in range(num_samples):
        for j in range(1, dimension + 1):
            phi[i].append(phi[i][j] ** 2)
            t = math.sqrt(abs(phi[i][j]))
            if phi[i][j] < 0:
                phi[i].append(-t)
            else:
                phi[i].append(t)

    phi = np.array(phi).T

    return phi

#generate k-folds dataset
def generate_validation_set(sample_phi, sample_y):
    sample_phi = sample_phi.T
    (N, d) = sample_phi.shape
    print (N, d)
    new_sample = [[] for _ in range(N)]
    for i in range(N):
        new_sample[i].append(sample_y[i])        
        for j in range(d):
            new_sample[i].append(sample_phi[i][j])
        
        
    
    #=============================================
    new_sample.sort()    
    N = N - N % num_fold
    new_sample = new_sample[:N]
    #==============================================    
    n_fold = int(N/num_fold)
    new_sample_phi, new_sample_y = [], []
    new_gt_phi, new_gt_y = [], []
    for fold_id in range(num_fold):
        big_phi_fold, big_y_fold = [], []
        one_phi_fold, one_y_fold = [], []
        for item_id in range(N):
            if item_id % num_fold == fold_id:
                one_phi_fold.append(new_sample[item_id][1:])
                one_y_fold.append(new_sample[item_id][0])
            else:
                big_phi_fold.append(new_sample[item_id][1:])
                big_y_fold.append(new_sample[item_id][0])

        new_sample_phi.append(big_phi_fold)
        new_sample_y.append(big_y_fold)
        new_gt_phi.append(one_phi_fold)
        new_gt_y.append(one_y_fold)


    new_sample_phi = np.array(new_sample_phi)
    new_sample_y = np.array(new_sample_y)
    new_gt_phi = np.array(new_gt_phi)
    new_gt_y = np.array(new_gt_y)    
    print (new_sample_phi.shape)
    print (new_sample_y.shape)
    print (new_gt_phi.shape)
    print (new_gt_y.shape)
    return new_sample_phi, new_sample_y, new_gt_phi, new_gt_y



def compute_mean_squared_error(pred_y, gt_y):
    N = pred_y.shape[0]
    return math.sqrt(sum([(pred_y[i] - gt_y[i]) ** 2 for i in range(N)])/N)

def compute_mean_absolute_error(pred_y, gt_y):
    N = pred_y.shape[0]
    return math.sqrt(sum([abs(pred_y[i] - gt_y[i]) for i in range(N)])/N)

def visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path, std=None):
    plt.plot(sample_x, sample_y, 'ro', label='samples')
    plt.plot(gt_x, gt_y, 'b.', label='ground truth')
    plt.plot(gt_x, pred_y, 'g.', label='predicted results')
    if std is not None:
        plt.errorbar(gt_x, pred_y, yerr=std, color = '#297083', ls = 'none', lw = 2, capthick = 2)

    plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.title(title)
    plt.savefig(img_path, dpi=100)
    plt.close('all')


def visualize_people_counting(gt_y, pred_y, title, img_path):
    y_min = min(min(gt_y), min(pred_y))
    y_max = max(max(gt_y), max(pred_y))
    x = np.linspace(y_min, y_max, 100)
    y = x
    plt.plot(gt_y, pred_y, 'b.')
    plt.plot(x, y, 'r')
    plt.xlabel('True counts')
    plt.ylabel('Predictions')

    plt.title(title)
    plt.savefig(img_path, dpi=100)
    plt.close('all')