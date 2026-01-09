import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
from src.data_generator import genrate_dataset
from src.preprocess import preprocess_bernoulli
from src.bernoulli_nb import train_bernoulli_nb
from src.gaussian_nb import train_gaussian_nb
from src.multinomial_nb import train_multinomial_nb
from src.evaluate import evaluate_model
from src.visualize import plot_accuracy

DATA_PATH = "data/dataset.csv"
genrate_dataset(DATA_PATH)
df = pd.read_csv(DATA_PATH)
X = df.drop("personality",axis=1).values
y = df["personality"].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

metrics = {}

#Gaussian
g_scalar = StandardScaler()
Xg_train = g_scalar.fit_transform(X_train)
Xg_test = g_scalar.fit_transform(X_test)
gnb = train_gaussian_nb(Xg_train,y_train)
metrics["GaussianNB"] = evaluate_model(gnb,Xg_test,y_test)

#Multinomial
m_scalar = MinMaxScaler()
Xm_train = (m_scalar.fit_transform(X_train)*5).round()
Xm_test = (m_scalar.fit_transform(X_test)*5).round()
mnb = train_multinomial_nb(Xm_train,y_train)
metrics["MultinomialNB"] = evaluate_model(mnb,Xm_test,y_test)

#Bernoulli 

Xb_train = preprocess_bernoulli(X_train)
Xb_test = preprocess_bernoulli(X_test)
bnb = train_bernoulli_nb(Xb_train,y_train)
metrics["BernoulliNB"] = evaluate_model(bnb,Xb_test,y_test)

with open("results/metrics.json","w") as f :
    json.dump(metrics,f,indent=4)

plot_accuracy(metrics)