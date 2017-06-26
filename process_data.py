#import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def load_processed_data_test():
    df = pd.read_csv("C:\\Users\\tati\\data.csv", sep=',')

    X = df[['got','great','wat','free','text','txt','win','already','dun','say','around','dont','think','back','hey','like',
    'now','send','still','even','friends','per','call','claim','customer','prize','mobile','gonna','home','ive','soon','today',
    'tonight','want','cash','reply','urgent','week','won','help','right','take','will','wont','message','next','make','name','yes',
    'feel','thats','way','miss','going','try','first','lor','can','meet','getting','just','lol','really','always','love','amp','ill',
    'know','let','work','sure','wait','yeah','anything','tell','please','thanks','msg','see','pls','need','nokia','tomorrow','hope',
    'ltgt','well','didnt','get','ask','cant','time','morning','place','give','happy','sorry','new','find','year','later','pick','good',
    'come','nice','said','day','money','babe','something','waiting','much','stop','one','late','night','someone','guaranteed','service',
    'buy','box','yet','youre','dear','life','people','cos','things','contact','last','went','sent','chat','gud','thk','keep','also','coming',
    'every','told','sleep','care','mins','phone','number','wish','leave','thing','many','wan','Common_Word_Count','Word_Count']]
    Y = pd.to_numeric(df['resposta'])


# tratando variáveis categóricas
    category_map = {}
    category_list = ['got','great','wat','free','text','txt','win','already','dun','say','around','dont','think','back','hey','like',
    'now','send','still','even','friends','per','call','claim','customer','prize','mobile','gonna','home','ive','soon','today',
    'tonight','want','cash','reply','urgent','week','won','help','right','take','will','wont','message','next','make','name','yes',
    'feel','thats','way','miss','going','try','first','lor','can','meet','getting','just','lol','really','always','love','amp','ill',
    'know','let','work','sure','wait','yeah','anything','tell','please','thanks','msg','see','pls','need','nokia','tomorrow','hope',
    'ltgt','well','didnt','get','ask','cant','time','morning','place','give','happy','sorry','new','find','year','later','pick','good',
    'come','nice','said','day','money','babe','something','waiting','much','stop','one','late','night','someone','guaranteed','service',
    'buy','box','yet','youre','dear','life','people','cos','things','contact','last','went','sent','chat','gud','thk','keep','also','coming',
    'every','told','sleep','care','mins','phone','number','wish','leave','thing','many','wan']

    # for each category transform into numbers
    for cat in category_list:
        encoder = LabelEncoder()
        X[cat] = encoder.fit_transform(X[cat]) # fitting this category and trasnforming the column to indexes
        category_map[cat] = encoder.classes_   # saving the indexes to know how to go back from index->category

# gerando matriz de OneHotEncoder
    # gerando matriz de OneHotEncoder
    categorical_indexes=[]
    for cat in category_list:
        categorical_indexes.append(X.columns.get_loc(cat))
    one_hot_encoder = OneHotEncoder(categorical_features = categorical_indexes)
    print(X.head())
    print(Y.head())
    return X, Y

if __name__ == '__main__':
    load_processed_data_test()
