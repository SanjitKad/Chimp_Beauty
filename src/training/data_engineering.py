import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from numpy import dot
from numpy.linalg import norm
from ast import literal_eval

class DataEngineering:

    def load_data(self):
        product_information_df = pd.read_excel('files/Makeup_Products_Metadata.xlsx')
        customer_review_df = pd.read_excel('files/User_review_data.xlsx')
        return [product_information_df, customer_review_df]

    def compress_product_information(self, product_information_df):
        required_fields = ['Product ID','Product Category', 'Product Brand', 'Product Name', 'Product Price [SEK]', 'Product Tags']
        reduced_product_information_df = product_information_df[required_fields]
        reduced_product_information_df['Product Tags'] = reduced_product_information_df['Product Tags'].str.replace('\d+', '')
        return reduced_product_information_df

    def cleanup_string(self, string):
        string = str(string)
        string = string.replace('Null','')
        string = string.replace('nan','')
        return string

    def generate_vocabulary(self, dataframe, column):
        results = set()
        dataframe[column].str.lower().str.split().apply(results.update)
        return list(results)

    def vectorize_product_information(self, reduced_product_information_df):

        product_category_vocabulary = self.generate_vocabulary(reduced_product_information_df,'Product Category')
        product_category_vectorizer = CountVectorizer(ngram_range=(1, 1), vocabulary=product_category_vocabulary)

        product_brand_vocabulary = self.generate_vocabulary(reduced_product_information_df, 'Product Brand')
        product_brand_vectorizer = CountVectorizer(ngram_range=(1, 1), vocabulary=product_brand_vocabulary)

        #product_category_vocabulary = generate_vocabulary(reduced_product_information_df, 'Product Tags')
        #product_tags_vectorizer =CountVectorizer(ngram_range=(1,1))

        for count, row in reduced_product_information_df.iterrows():
                reduced_product_information_df.at[count,'Product Category'] = product_category_vectorizer.fit_transform([self.cleanup_string(row['Product Category'])]).tocoo().col.tolist()
                reduced_product_information_df.at[count,'Product Brand'] = product_brand_vectorizer.fit_transform([self.cleanup_string(row['Product Brand'])]).tocoo().col.tolist()
                #reduced_product_information_df.at[count,'Product Tags'] = product_tags_vectorizer.fit_transform([cleanup_string(row['Product Tags'])])
        return reduced_product_information_df


    # def calculate_similarity_score(self, product_row, check_row):
    #     check_category = check_row['Product Category'].toarray()
    #     product_category = product_row['Product Category'].iloc[0].toarray()
    #     category_similarity = dot(check_category, product_category.T)/(norm(check_category)*norm(product_category))
    #
    #     check_brand = check_row['Product Brand'].toarray()
    #     product_brand = product_row['Product Brand'].iloc[0].toarray()
    #     brand_similarity = dot(check_brand, product_brand.T) / (norm(check_brand) * norm(product_brand))
    #
    #     return (category_similarity[0][0]+brand_similarity[0][0])/2

    def calculate_similarity_score(self, product_row, check_row):
        check_category = np.array(literal_eval(check_row['Product Category']))
        check_category = np.pad(check_category, (0, 10-len(check_category)), 'constant')

        product_category = np.array(literal_eval(product_row['Product Category'].iloc[0]))
        product_category = np.pad(product_category, (0, 10 - len(product_category)), 'constant')

        category_similarity = dot(check_category, product_category.T) / (norm(check_category) * norm(product_category))

        check_brand = np.array(literal_eval(check_row['Product Brand']))
        check_brand = np.pad(check_brand, (0, 10 - len(check_brand)), 'constant')

        product_brand = np.array(literal_eval(product_row['Product Brand'].iloc[0]))
        product_brand = np.pad(product_brand, (0, 10 - len(product_brand)), 'constant')

        brand_similarity = dot(check_brand, product_brand.T) / (norm(check_brand) * norm(product_brand))

        return (category_similarity + brand_similarity) / 2

    def suggest_products(self,user_name):
        vectorized_df = pd.read_csv('vectorized_product_information.csv')
        customer_df = pd.read_excel('files/User_review_data.xlsx')

        suggested_products = []
        user_row = customer_df.loc[customer_df['User'] == user_name]
        user_row.drop(['User'], axis=1)
        product_codes = list(user_row.apply(lambda row: row[row != 0].index.tolist(), axis=1))
        product_codes = product_codes[0][1:len(product_codes[0]) - 1]

        product_scores = list(user_row.apply(lambda row: row[row != 0].tolist(), axis=1))
        product_scores = product_scores[0][1:len(product_scores[0]) - 1]

        for product_code in product_codes:
            product_row = vectorized_df.loc[vectorized_df['Product ID'] == product_code]
            for count, check_row in vectorized_df.iterrows():
                similarity_score = self.calculate_similarity_score(product_row, check_row)
                if similarity_score > 0.993 and similarity_score != 1.0:
                    suggested_products.append(product_code)

        return suggested_products


data_engineering = DataEngineering()
# [product_information_df, customer_review_df] = data_engineering.load_data()
# reduced_product_information_df = data_engineering.compress_product_information(product_information_df)
# vectorized_df = data_engineering.vectorize_product_information(reduced_product_information_df)
# vectorized_df.to_csv('vectorized_product_information.csv', sep=',', index=False, encoding='utf-8')


user_name = 'Kevin'
product_suggestions = data_engineering.suggest_products(user_name)

