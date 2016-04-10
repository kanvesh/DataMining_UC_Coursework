import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC   


# The script expects a csv file as input which has the comments under the 'viol_observation' column and a corresponding risk category under the 'risk_cat' column
# In the current example, the risk categories are basic, intermediate and high priority
# For Washington we have a score in place of risk category. We have converted score <=5 as Basic, score 10 and 15 as Intermediate and score above 20 as high priority

df = pd.read_csv('Terrene_Florida_Washington_Comments.csv')  #File that contains comments and categories

print df.head() # Print the first few rows of the imported file

# Converting the comments to the TF-IDF matrix

mat = TfidfVectorizer(analyzer='word', ngram_range= (2,5), stop_words='english')
vec = mat.fit(df['viol_observation'])
trans = mat.fit_transform(df['viol_observation'])

# Print the features out

features = vec.get_feature_names()
pd.DataFrame(features).to_csv('features.csv', encoding='utf-8')


# Training a linear SVM model with the risk category as 'y' and the tfidf matrix as 'x'

svm  = LinearSVC()
model = svm.fit(trans,df['risk_cat'])

# Print the coefficients out

coef = pd.DataFrame(model.coef_).transpose()
coef.columns = columns = list(model.classes_)
coef.to_csv('coef.csv')



# Printing the features and their coefficient values indicating the strength of their relation
feat_coef = pd.concat([pd.DataFrame(features), coef], axis=1)
feat_coef.to_csv('Features_Coefficients.csv', encoding='utf-8')

