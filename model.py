import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class Recommendation:

    def __init__(self):
        self.data = pickle.load(open('pickle/sentiment_data.pkl', 'rb'))
        self.user_final_rating = pickle.load(open('pickle/item_based_recommendation.pkl', 'rb'))
        self.model = pickle.load(open('pickle/sentiment_analysis_model.pkl', 'rb'))
        self.raw_data = pd.read_csv("data/sample30.csv")
        self.data = pd.concat([self.raw_data[['id', 'name', 'brand', 'categories', 'manufacturer']], self.data], axis=1)

    def getTopProducts(self, user):

        if user in self.raw_data.values:
            items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
            features = pickle.load(open('pickle/tfid_features.pkl', 'rb'))
            vectorizer = TfidfVectorizer(vocabulary=features)
            temp = self.data[self.data.id.isin(items)]
            X = vectorizer.fit_transform(temp['Review'])
            temp = temp[['id']]
            temp['prediction'] = self.model.predict(X)
            temp['prediction'] = temp['prediction'].map({'Positive': 1, 'Negative': 0})
            temp = temp.groupby('id').sum()
            temp['positive_percent'] = temp.apply(lambda x: x['prediction']/sum(x), axis=1)
            final_list = temp.sort_values('positive_percent', ascending=False).iloc[:5, :].index
            final_data = self.data[self.data.id.isin(final_list)][['name']].drop_duplicates()
            final_data = final_data.rename({"name": "Name Of Product"}, axis='columns').to_html(index=False)
            return final_data
        else:
            return ""