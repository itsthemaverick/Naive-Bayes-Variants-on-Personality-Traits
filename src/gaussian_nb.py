from sklearn.naive_bayes import GaussianNB

def train_gaussian_nb(X,y):
    model = GaussianNB()
    model.fit(X,y)
    return model