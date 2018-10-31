from params import *




def generate_polynomial_features(x, polynomial_degree=5):
    phi = [x ** 0]
    for k in range(1, polynomial_degree + 1):
        phi = np.append(phi, [x ** k], axis=0)
    return phi

def compute_mean_squared_error(pred_y, gt_y):
    N = pred_y.shape[0]
    return math.sqrt(sum([(pred_y[i] - gt_y[i]) ** 2 for i in range(N)])/N)

def visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path, std=None):
    plt.plot(sample_x, sample_y, 'ro', label='samples')
    plt.plot(gt_x, gt_y, 'b.', label='ground truth')
    plt.plot(gt_x, pred_y, 'g.', label='predicted results')
    if std is not None:
        plt.errorbar(gt_x, pred_y, yerr=std, color = '#297083', ls = 'none', lw = 2, capthick = 2)

    plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    # plt.xlabel('Months')
    # plt.ylabel('Books Read')
    plt.title(title)
    plt.savefig(img_path, dpi=100)
    plt.close('all')
