import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommendation_system(user_id, bank_data, similarity):
    user_index = bank_data[bank_data['user_id'] == int(user_id)].index[0]
    similar_users_indices = np.argsort(similarity[user_index])[::-1][1:]
    
    similar_products_info = []
    
    for index in similar_users_indices[:2]:
        similar_user_product_names = bank_data.loc[index, 'Products']
        similarity_score = similarity[user_index][index]
        similar_products_info.append((similar_user_product_names, similarity_score))
    
    return similar_products_info

def main():
    # Read the CSV file and preprocess the data
    bank_data = pd.read_csv('BankCustomerData.csv')
    selected_features = ['job', 'marital', 'education', 'balance', 'housing_loan', 'vehicle_loan', 'duration']
    for feature in selected_features:
        bank_data[feature] = bank_data[feature].fillna('')
    bank_data['balance'] = bank_data['balance'].astype(str)
    bank_data['duration'] = bank_data['duration'].astype(str)
    combined_features = (
        bank_data['job'].astype(str) + ' ' +
        bank_data['marital'] + ' ' +
        bank_data['education'] + ' ' +
        bank_data['balance'] + ' ' +
        bank_data['housing_loan'] + ' ' +
        bank_data['vehicle_loan'] + ' ' +
        bank_data['duration']
    )
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)

    # Streamlit UI
    st.title("Product Recommendation System")

    # Dropdown to select user ID
    selected_user_id = st.selectbox("Select a user ID", bank_data['user_id'].values)

    if st.button("Generate Recommendations"):
        recommendations = recommendation_system(selected_user_id, bank_data, similarity)
        
        # Display user's existing products in a table format
        user_row = bank_data[bank_data['user_id'] == selected_user_id]
        existing_products = user_row['Products'].values[0]
        existing_products_list = existing_products.split(',')
        
        # Display recommended products in a table format
        recommended_products_set = set()  # To store recommended products without duplicates
        for product_info in recommendations:
            product_name, similarity_score = product_info
            recommended_products_set.update(product_name.split(','))
        # Remove duplicates from recommended products that are already in existing products
        recommended_products_set -= set(existing_products_list)
        st.write(recommended_products_set)
        recommended_products_str = ', '.join(recommended_products_set)
        st.write(recommended_products_str)
        recommended_products_list = recommended_products_str.split(',')
        st.write(recommended_products_list)
        # Create dataframes for existing and recommended products
        existing_table = pd.DataFrame(existing_products_list, columns=["Existing Products"])
        recommended_table = pd.DataFrame(recommended_products_list, columns=["Recommended Products"])
        
        # Display tables side by side
        st.write("User's Existing Products vs. Recommended Products:")
        st.write(pd.concat([existing_table, recommended_table], axis=1))
        
if __name__=="__main__":
    main()
