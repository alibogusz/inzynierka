from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, KFold
import time

from data_proccessing import *

X_train = GetData().X_train
X_test = GetData().X_test
y_train = GetData().y_train


def test_split(X_train, y_train):

    sizes = np.arange(0.1, 0.4, 0.02)
    split_score = []
    for i in sizes:
        train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size = i, random_state = 1)

        hist = HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.1, max_iter=500, max_leaf_nodes=31, max_depth=5)
        hist.fit(train_x, train_y)

        Y_pred = hist.predict(val_x)

        y_pred_hist = sc.inverse_transform([Y_pred])
        y_true = sc.inverse_transform([val_y])

        score_hist = hist.score(train_x, train_y)  
        # print("Training score: ", score_)
        split_score.append(score_hist)

    plt.plot(sizes, split_score, '-o')
    plt.title('HistGradientBoost score by test size')
    plt.grid()
    plt.ylabel('score')
    plt.xlabel('test size')
    plt.show()

    x_ax = range(len(y_true[0][:100]))

    plt.figure(figsize=(10,5))
    plt.scatter(x_ax, y_true[0][:100], label="original")
    plt.scatter(x_ax, y_pred_hist[0][:100], label="predicted")
    plt.title("First 100 prices predicted with changing test size")
    plt.grid()
    plt.legend()
    plt.show()


# def evaluate_models(X_train, y_train):

#     models = dict()
#     for i in [10, 50, 100, 150, 200, 255]:
#         models[str(i)] = HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.1, max_iter=500, max_leaf_nodes=31, max_depth=5, max_bins=i)


#     # define the evaluation procedure
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#     # evaluate the models and store results
#     results, names = list(), list()
#     for name, model in models.items():
#         # evaluate the model and collect the scores
#         scores = cross_val_score(model, X_train, y_train, scoring='accuracy')
#         # stores the results
#         results.append(scores)
#         names.append(name)

#     # report performance along the way
#     print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
#     # plot model performance for comparison
#     plt.boxplot(results, labels=names, showmeans=True)
#     plt.show()


def test_loss(X_train, y_train):

    losses = ['squared_error', 'absolute_error', 'quantile']
    loss_score = []
    for i in losses:
        train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size = 0.33, random_state = 1)

        hist = HistGradientBoostingRegressor(loss=i, quantile=0.9, learning_rate=0.1, max_iter=1000, max_leaf_nodes=31, max_depth=7)
        hist.fit(train_x, train_y)

        Y_pred = hist.predict(val_x)

        # y_pred_hist = sc.inverse_transform([Y_pred])
        # y_true = sc.inverse_transform([val_y])

        score_hist = hist.score(train_x, train_y)  
        print("Training score: ", score_hist)
        loss_score.append(score_hist)

    plt.bar(losses, loss_score)
    plt.title('HistGradientBoost score by loss function')
    plt.grid()
    plt.ylabel('score')
    plt.show()

    # x_ax = range(len(y_true[0][:100]))

    # plt.figure(figsize=(10,5))
    # plt.scatter(x_ax, y_true[0][:100], label="original")
    # plt.scatter(x_ax, y_pred_hist[0][:100], label="predicted")
    # plt.title("First 100 prices predicted with changing test size")
    # plt.grid()
    # plt.legend()
    # plt.show()


def test_max_iter(X_train, y_train):

    N = np.arange(100, 2500, 200)
    iter_score = []
    for n in N:
        train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size = 0.33, random_state = 1)

        hist = HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.1, max_iter=n, max_leaf_nodes=31, max_depth=7)
        hist.fit(train_x, train_y)

        Y_pred = hist.predict(val_x)

        y_pred_hist = sc.inverse_transform([Y_pred])
        y_true = sc.inverse_transform([val_y])

        score_hist = hist.score(train_x, train_y)  
        # print("Training score: ", score_)
        iter_score.append(score_hist)

    plt.plot(N, iter_score, '-o')
    plt.title('HistGradientBoost score by iteration')
    plt.grid()
    plt.ylabel('score')
    plt.xlabel('iterations')
    plt.show()

    # x_ax = range(len(y_true[0][:100]))

    # plt.figure(figsize=(10,5))
    # plt.scatter(x_ax, y_true[0][:100], label="original")
    # plt.scatter(x_ax, y_pred_hist[0][:100], label="predicted")
    # plt.title("First 100 prices predicted with changing test size")
    # plt.grid()
    # plt.legend()
    # plt.show()

def test_max_leaf(X_train, y_train):

    N = np.arange(2, 31, 2)
    leaf_score = []
    for i in N:
        train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size = 0.33, random_state = 1)

        hist = HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.1, max_iter=1500, max_leaf_nodes=i, max_depth=7)
        hist.fit(train_x, train_y)

        Y_pred = hist.predict(val_x)

        y_pred_hist = sc.inverse_transform([Y_pred])
        y_true = sc.inverse_transform([val_y])

        score_hist = hist.score(train_x, train_y)  
        # print("Training score: ", score_)
        leaf_score.append(score_hist)

    plt.plot(N, leaf_score, '-o')
    plt.title('HistGradientBoost score by number of leaf nodes')
    plt.grid()
    plt.ylabel('score')
    plt.xlabel('leaf nodes')
    plt.show()


def test_max_depth(X_train, y_train):

    N = np.arange(1, 10, 1)
    depth_score = []
    for n in N:
        train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size = 0.33, random_state = 1)

        hist = HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.1, max_iter=1500, max_leaf_nodes=31, max_depth=n)
        hist.fit(train_x, train_y)

        Y_pred = hist.predict(val_x)

        y_pred_hist = sc.inverse_transform([Y_pred])
        y_true = sc.inverse_transform([val_y])

        score_hist = hist.score(train_x, train_y)  
        # print("Training score: ", score_)
        depth_score.append(score_hist)

    plt.plot(N, depth_score, '-o')
    plt.title('HistGradientBoost score by tree max depth')
    plt.grid()
    plt.ylabel('score')
    plt.xlabel('max depth')
    plt.show()

def predict_final(X_train, y_train, X_test, n, d):

    hist = HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.1, max_iter=n, max_leaf_nodes=31, max_depth=d)

    hist.fit(X_train, y_train)

    y_ = hist.predict(X_test)

    y_pred = sc.inverse_transform([y_])

    score_ = hist.score(X_train, y_train)  
    print("Training score: ", score_)

    # return score_
    solution = pd.DataFrame(y_pred[0], columns = ['PRICE'])
    print(solution)
    solution.to_excel('HistGradBoost_output.xlsx')

if __name__ == "__main__":
    # test_split(X_train, y_train)
    # evaluate_models(X_train, y_train)
    # test_loss(X_train, y_train)
    # test_max_iter(X_train, y_train)
    # test_max_leaf(X_train, y_train)
    # test_max_depth(X_train, y_train)
    predict_final(X_train, y_train, X_test, 1500, 7)

#     times = []
#     scores = []
#     I = np.arange(100, 3500, 200)
#     for i in I:
#         start = time.time()
#         score = predict_final(X_train, y_train, X_test, i, 7)
#         stop = time.time()
#         times.append(stop-start)
#         scores.append(score)


# fig, ax1 = plt.subplots()
# ax1.plot(I, times, color='blue', marker='o', label='run time')
# ax1.grid()
# ax1.set_title('HistGradientBoost run time and scores by itterations')
# ax1.set_xlabel('itteration')
# ax1.set_ylabel('time', color='blue')

# ax2=ax1.twinx()
# ax2.plot(I, scores, color='orange', marker='o', label='score')
# ax2.set_ylabel("score", color='orange')
# plt.show()