from sklearn.naive_bayes import BernoulliNB

def train_bernoulli_nb(X,y):
    model = BernoulliNB()
    model.fit(X,y)
    return model
