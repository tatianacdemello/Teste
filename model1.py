from sklearn import *
from process_data import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def train_test_model(name, m1, X_train, Y_train, X_test, Y_test):
#    name, m1, X_train, Y_train, X_test, Y_test = args
    print(m1)
    #parameters = {'C':[1, 10], 'epsilon':[0.01,0.05,0.1],'tol':[0.01, 0.001, 0.1,0.005],'max_iter':[1000,5000,10000]}
    #teste = GridSearchCV(m1,parameters,n_jobs=-1,cv = 3)
    #teste = ml
    #teste.fit(X_train, Y_train)
    m1.fit(X_train, Y_train)

    y_pred = m1.predict(X_test)
    results = open('out_%s' % (name), 'w')
    results.write("R2 Score: %.2f\n" % (metrics.r2_score(Y_test, y_pred)))
    results.write("Explained Variance Score: %.2f\n" % (metrics.explained_variance_score(Y_test, y_pred)))
    results.write("Mean Absolute Error: %.2f\n" % (metrics.mean_absolute_error(Y_test, y_pred)))
    results.write("Mean Squared Error: %.2f\n" % (metrics.mean_squared_error(Y_test, y_pred)))
    results.write("Median Absolute Error: %.2f\n" % (metrics.median_absolute_error(Y_test, y_pred)))
    results.close()
    ex = pd.DataFrame(columns=['real', 'pred'])
    ex['real'] = Y_test
    ex['pred'] = y_pred
    ex.to_csv('%s.csv' % name, index=False, sep=";")
    externals.joblib.dump(m1, "%s.pkl" % name, compress=9)


if __name__ == '__main__':
    X, Y = load_processed_data_test()
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.33, random_state=42)
    #m1 = svm.LinearSVR()
    #ml = GradientBoostingClassifier(n_estimators=5000, learning_rate=2**(-9.5), max_features='log2', max_depth=7, random_state=1, verbose=1)
    m1 = LogisticRegression(C=10.0, multi_class='multinomial', solver='lbfgs', verbose=1)
    name = "LogisticRegression"
    #externals.joblib.dump((X, Y, X_train, Y_train, X_test, Y_test), "dataset.pkl", compress=9)
    train_test_model(name, m1, X_train, Y_train, X_test, Y_test)
