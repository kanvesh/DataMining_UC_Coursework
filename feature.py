import pandas as pd
#import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC   

df = pd.read_csv('Terrene_comments_cleaned_small.csv')
#df = df[1:10000,]

print df.head()

mat = TfidfVectorizer(max_df=0.5, min_df=0.001, analyzer='word', ngram_range= (1,5), stop_words='english')
vec = mat.fit(df['viol_observation'])
trans = mat.fit_transform(df['viol_observation'])

print vec.get_feature_names()

print type(trans)
print trans.shape

svm  = LinearSVC()

model = svm.fit(trans,df['risk_cat'])

pd.DataFrame(model.coef_).to_csv('coef.csv')


