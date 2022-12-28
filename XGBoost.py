from sklearn.metrics import accuracy_score, explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

from xgboost import XGBRegressor

from data_proccessing import *

X_train = GetData().X_train
X_test = GetData().X_test
y_train = GetData().y_train


def test_split(X_train, y_train):

    sizes = np.arange(0.1, 0.4, 0.02)

    split_score = []
    for i in sizes:
        train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size = i, random_state = 1)

        xgb=XGBRegressor(n_estimators=1000, max_depth=7, learning_rate = 0.1)

        xgb.fit(train_x, train_y)

        y_ = xgb.predict(val_x)

        y_pred = sc.inverse_transform([y_])
        y_true = sc.inverse_transform([val_y])

        score_ = xgb.score(train_x, train_y)  
        # print("Training score: ", score_)
        split_score.append(score_)


    plt.plot(sizes, split_score, '-o')
    plt.title('XGBoost score by test_size value in train_test_split function')
    plt.grid()
    plt.ylabel('score')
    plt.xlabel('test size')
    plt.show()

    x_ax = range(len(y_true[0][:100]))

    plt.figure(figsize=(10,5))
    plt.scatter(x_ax, y_true[0][:100], label="original")
    plt.scatter(x_ax, y_pred[0][:100], label="predicted")
    plt.title("First 100 prices predicted with changing test size")
    plt.grid()
    plt.legend()
    plt.show()


def test_n_estimators(X_train, y_train, X_test):

    train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size = 0.33, random_state = 1)
    
    skorr = []
    N = np.arange(100, 1500, 200)
    for n in N:
        xgb=XGBRegressor(n_estimators=n, max_depth=6, learning_rate = 0.1)

        xgb.fit(train_x, train_y)

        y_ = xgb.predict(val_x)

        y_pred = sc.inverse_transform([y_])
        y_true = sc.inverse_transform([val_y])

        score_ = xgb.score(train_x, train_y)  
        # print("Training score: ", score_)
        skorr.append(score_)

    plt.plot(N, skorr, '-o')
    plt.title('XGBoost score by number of estimators')
    plt.grid()
    plt.ylabel('score')
    plt.xlabel('n estimators')
    plt.show()

    x_ax = range(len(y_true[0][:100]))

    plt.figure(figsize=(10,5))
    plt.scatter(x_ax, y_true[0][:100], label="original")
    plt.scatter(x_ax, y_pred[0][:100], label="predicted")
    plt.title("First 100 prices predicted with changing number of estimators")
    plt.grid()
    plt.legend()
    plt.show()


def test_max_depth(X_train, y_train, X_test):

    train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size = 0.33, random_state = 123)

    skorr = []
    D = np.arange(1, 10)
    for d in D:
        xgb=XGBRegressor(n_estimators=1000, max_depth=d, learning_rate = 0.1)

        xgb.fit(train_x, train_y)

        y_ = xgb.predict(val_x)

        y_pred = sc.inverse_transform([y_])
        y_true = sc.inverse_transform([val_y])

        score_ = xgb.score(train_x, train_y)  
        # print("Training score: ", score_)
        skorr.append(score_)

    plt.plot(D, skorr, '-o')
    plt.title('XGBoost score by max depth')
    plt.grid()
    plt.ylabel('score')
    plt.xlabel('max depth')
    plt.show()

    x_ax = range(len(y_true[0][:100]))

    #calculate equation for quadratic trendline
    z1 = np.polyfit(x_ax, y_true[0][:100], 2)
    z2 = np.polyfit(x_ax, y_pred[0][:100], 2)
    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)

    #add trendline to plot
    plt.figure(figsize=(10,5))
    plt.scatter(x_ax, y_true[0][:100], label="original")
    plt.scatter(x_ax, y_pred[0][:100], label="predicted")
    plt.plot(x_ax, p1(x_ax))
    plt.plot(x_ax, p2(x_ax))
    plt.title("First 100 prices predicted with changing max depth")
    plt.grid()
    plt.legend()
    plt.show()


def predict_final(X_train, y_train, X_test, n, d):

    xgb=XGBRegressor(n_estimators=n, max_depth=d, learning_rate = 0.1)

    xgb.fit(X_train, y_train)

    y_ = xgb.predict(X_test)

    y_pred = sc.inverse_transform([y_])
    # y_true = sc.inverse_transform([val_y])

    score_ = xgb.score(X_train, y_train)  
    print("Training score: ", score_)

    solution = pd.DataFrame(y_pred[0], columns = ['PRICE'])
    print(solution)
    solution.to_excel('XGB_output.xlsx')

# # scores = cross_val_score(xgb, train_x, train_y, cv=100)
# # print("Mean cross-validation score: {0} with deviation {1}".format(scores.mean(), scores.std()))

if __name__ == "__main__":
    # test_split(X_train, y_train)
    test_n_estimators(X_train, y_train, X_test)
    # test_max_depth(X_train, y_train, X_test)
    # predict_final(X_train, y_train, X_test, 1000, 7)