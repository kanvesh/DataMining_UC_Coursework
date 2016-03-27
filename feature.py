import pandas as pd
#import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC   

df = pd.read_csv('Terrene_comments_cleaned.csv')
#df = df.iloc[1:10000,]

print df.head()

mat = TfidfVectorizer(max_df=0.5, min_df=0.001, analyzer='word', ngram_range= (2,5), stop_words='english')
vec = mat.fit(df['viol_observation'])
trans = mat.fit_transform(df['viol_observation'])

features = vec.get_feature_names()
print type(features)
print len(vec.get_feature_names())

pd.DataFrame(features).to_csv('features.csv')

print type(trans)
print trans.shape

svm  = LinearSVC()

model = svm.fit(trans,df['risk_cat'])

coef = pd.DataFrame(model.coef_).transpose()

coef.to_csv('coef.csv')

feat_coef = pd.concat([pd.DataFrame(features), coef], axis=1)

feat_coef.to_csv('Features_Coefficients.csv')

