from sklearn.naive_bayes import MultinomialNB

def train_multinomial_nb(X,y):
    model = MultinomialNB()
    model.fit(X,y)
    return model
