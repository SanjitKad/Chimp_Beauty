from ast import literal_eval

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm


class SuggestProducts:

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

    def suggest_products(self, user_name):
        #loading ing vectorized products and user reviews
        vectorized_df = pd.read_csv('src/testing/files/vectorized_product_information.csv')
        customer_df = pd.read_csv('src/testing/files/User_review_data.csv')

        suggested_products = []
        user_row = customer_df.loc[customer_df['User'] == user_name]
        user_row.drop(['User'], axis=1)
        product_codes = list(user_row.apply(lambda row: row[row != 0].index.tolist(), axis=1))
        product_codes = product_codes[0][1:len(product_codes[0]) - 1]

        product_scores = list(user_row.apply(lambda row: row[row != 0].tolist(), axis=1))
        product_scores = product_scores[0][1:len(product_scores[0]) - 1]

        # looping through products to find similar ones.

        for product_code in product_codes:
            try:
                product_row = vectorized_df.loc[vectorized_df['Product ID'] == int(product_code)]
                for count, check_row in vectorized_df.iterrows():
                    similarity_score = self.calculate_similarity_score(product_row, check_row)
                    if similarity_score > 0.993 and similarity_score != 1.0:
                        suggested_products.append(product_code)
            except:
                continue

        # suggesting 10 similar products
        return suggested_products[0:10]