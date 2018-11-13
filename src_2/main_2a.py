from params import *
from estimators import *
from dataset import DataSet
from common import *

result_sub_path = result_path + 'part2a/'
#get data
sample_data = DataSet(cnt_sample_x_path, cnt_sample_y_path)
sample_x, sample_y = sample_data.x, sample_data.y
sample_phi = sample_x
print (sample_phi.shape)
print (sample_y.shape)

val_sample_phi, val_sample_y, val_gt_phi, val_gt_y \
                        = generate_validation_set(sample_phi, sample_y)

gt_data = DataSet(cnt_gt_x_path, cnt_gt_y_path)
gt_x, gt_y = gt_data.x, gt_data.y
gt_phi = gt_x


def deploy_model(model, _lambda=None):
    print ("--------------------------------------------------------")
    print (model.name)
    log = open(result_sub_path + model.name + '.txt', 'w')
    img_path = result_sub_path + model.name + '.png'

    if _lambda == None:
        model.fit(sample_phi, sample_y)
    else:
        model.fit(sample_phi, sample_y, _lambda)
        print ("Lambda:", _lambda)
    
    pred_std=None
    if model.name == "BayesianRegression":
        pred_sample_y, pred_sample_std = model.predict(sample_phi)
        pred_y, pred_std = model.predict(gt_phi)
    else:
        pred_sample_y = model.predict(sample_phi)
        pred_y = model.predict(gt_phi)

    sample_error = compute_mean_squared_error(pred_sample_y, sample_y)
    error = compute_mean_squared_error(pred_y, gt_y)

    print ("Training MSE: " + str(sample_error))
    print ("Testing MSE: " + str(error))
    log.write("Lambda: " + str(_lambda) + '\n')
    log.write("Training MSE: " + str(sample_error) + '\n')
    log.write("Testing MSE: " + str(error) + '\n')

    sample_error = compute_mean_absolute_error(pred_sample_y, sample_y)
    error = compute_mean_absolute_error(pred_y, gt_y)

    print ("Training MAE: " + str(sample_error))
    print ("Testing MAE: " + str(error))
    log.write("Training MAE: " + str(sample_error) + '\n')
    log.write("Testing MAE: " + str(error) + '\n')

    visualize_people_counting(gt_y, pred_y, model.name, img_path)


def estimate_k_fold_error(model, _lambda):
    sample_error = 0

    if model.name == "RegularizedLeastSquares":
        for fold_id in range(num_fold):
            model.fit(val_sample_phi[fold_id].T, val_sample_y[fold_id], _lambda)
            pred_y = model.predict(val_gt_phi[fold_id].T)
            sample_error += compute_mean_squared_error(pred_y, val_gt_y[fold_id])

    return sample_error / num_fold


def estimate_best_lambda(model):
    val_error = []
    for _lambda in candidate_hyperparams:
        val_error.append(estimate_k_fold_error(model=model, _lambda=_lambda))

    return candidate_hyperparams[val_error.index(min(val_error))]


#========================================= MAIN ====================================================
#LEAST SQUARES
deploy_model(model=LeastSquares())

#REGULARIZED LEAST SQUARES
best_lambda = estimate_best_lambda(RegularizedLeastSquares())
deploy_model(model=RegularizedLeastSquares(), _lambda=best_lambda)

#LASSO
best_lambda = estimate_best_lambda(Lasso())
deploy_model(model=Lasso(), _lambda=best_lambda)

#ROBUST REGRESSION
deploy_model(model=RobustRegression())

#BAYESIAN REGRESSION
best_lambda = estimate_best_lambda(BayesianRegression())
deploy_model(model=BayesianRegression(), _lambda=best_lambda)
