import numpy as np 
import pandas as pd 

def genrate_dataset(path:str,n_samples:int = 20000,random_state:int = 42 ):
    np.random.seed(random_state)

    extroversion = np.random.normal(loc=6.0,scale=1.5,size=n_samples)
    openness = extroversion*0.6 + np.random.normal(0,1.2,n_samples)
    conscientiousness = np.random.normal(5.5,1.3,n_samples)
    agreeableness = conscientiousness*0.4 + np.random.normal(0,1.1,n_samples)
    neuroticism = np.random.normal(5.5,1.4,n_samples) - extroversion*5


    X = np.column_stack([
        extroversion,
        openness,
        conscientiousness,
        agreeableness,
        neuroticism
    ])

    X = np.clip(X,0,10)

    latent_score = (0.4*extroversion + 0.3*openness + 0.2*agreeableness- 0.3*neuroticism + np.random.normal(0,1.0,n_samples) )

    probablities = 1/(1+np.exp(-latent_score))
    y = (np.random.rand(n_samples)< probablities).astype(int)

    df = pd.DataFrame(X,columns=[
        "extroversion",
        "openness",
        "conscientiousness",
        "agreeableness",
        "neuroticism"
    ])

    df["personality"] = y
    df.to_csv(path,index=False)

    