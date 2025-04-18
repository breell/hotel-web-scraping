import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as sklearn_split

# Title
st.set_page_config(layout="wide")
st.title('Stays Recommedation System')

# upload dataset
# file_upload = st.file_uploader('upload csv', type=['csv'])
df_rent = pd.read_csv('bukit-vista-airbnb(2).csv')

# st.subheader('Dataset preview')
# st.dataframe(df_rent.head())

# # pick hotel
# st.subheader('Bali & Yogyakarta Stays')
# stays_list = df_rent['name'].unique()
# selected_stays = st.selectbox('Choose a Stay', stays_list)
# st.write(f'You selected: **{selected_stays}**')

# selected_stays = st.selectbox('Select a stay for recommendation:', df_rent['name'].unique())
# # recommendation func
# if st.button('Show Recommendation'):
#     st.write('Generating Recommendations...')



#==== train Collaborative Filtering Model (SVD & RF)========
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_rent[['user_id', 'name', 'rating']], reader)
trainset, testset = surprise_split(data, test_size=0.2, random_state=42)

svd_model = SVD()
svd_model.fit(trainset)

features = ['property_category','bedrooms','bathrooms','city','state','area',
            'number_of_guests','price_value','currency','period','Amazing View',
            'Amazing pool','Beachfront','Golfing','Guest House','Island life',
            'Jungle View','Ocean view','Pool view','Residential','Rice paddy view','Surfing']
target = 'rating'

x = df_rent[features]
y = df_rent[target]
x = pd.get_dummies(x)

x_train, x_test, y_train, y_test = sklearn_split(x, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# streamlit UI
st.title('Rating Prediction & Hybrid Models Recommender')
user_ids = df_rent['user_id'].unique()
input_user_id = st.selectbox('Choose User ID', sorted(user_ids))

# show hotels that has been chosen before by user
st.subheader('stays that has been been chosen before by user')
user_data = df_rent[df_rent['user_id'] == input_user_id]

if not user_data.empty:
    for idx, row in user_data.iterrows():
        st.markdown(f'''
                    **Hotel:** {row['name']}  
                    **City:** {row['city']}, **Area:** {row['area']}  
                    **Guests:** {row['number_of_guests']}, **Price:** {row['price_value']} {row['currency']}  
                    **Bedrooms:** {row['bedrooms']}, **Bathrooms:** {row['bathrooms']}  
                    **Amenities:** Jungle View: {row['Jungle View']}, Surfing: {row['Surfing']}, Pool View: {row['Pool view']}, Golfing: {row['Golfing']}
                    ''')
        
    def hybrid_recommendation(user_id, top_n=5, alpha=0.5):
        all_hotels = df_rent['name'].unique()
        rated = df_rent[df_rent['user_id'] == user_id]['name'].unique()
        unseen = [hotel for hotel in all_hotels if hotel not in rated]

        hybrid_scores = []

        for hotel in unseen:
            svd_pred = svd_model.predict(user_id, hotel).est

            hotel_features = df_rent[df_rent['name'] == hotel][features]
            hotel_features_encoded = pd.get_dummies(hotel_features)
            hotel_features_encoded = hotel_features_encoded.reindex(columns=x.columns, fill_value=0)

            rf_pred = rf_model.predict(hotel_features_encoded)[0]
            score = alpha * svd_pred + (1 - alpha) * rf_pred
            hybrid_scores.append((hotel, score, svd_pred, rf_pred))

        sorted_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True) [:top_n]

        st.subheader(f'Top {top_n}) Recommendation for User ID {user_id}')
        for hotel, hybrid_score, svd_score, rf_score in sorted_scores:
            info = df_rent[df_rent['name'] == hotel].iloc[0]
            st.markdown(f'''
                        **Hotel:** {hotel}  
                        **City:** {info['city']}, **Area:** {info['area']}  
                        **Guests:** {info['number_of_guests']}, **Price:** {info['price_value']} {info['currency']}  
                        **Bedrooms:** {info['bedrooms']}, **Bathrooms:** {info['bathrooms']}  
                        **Amenities:** Jungle View: {info['Jungle View']}, Surfing: {info['Surfing']}, Pool View: {info['Pool view']}, Golfing: {info['Golfing']}  
                        **Hybrid Score:** {hybrid_score:.2f} | **SVD:** {svd_score:.2f} | **RF:** {rf_score:.2f}
                        ''')
            st.markdown("---")
# ========= content-based similarity ==========
    def show_content_based_recommendation(user_id, top_n=5):
        st.subheader(f'Top{top_n} Content-Based Recommednations for User ID{user_id}')
        valid_features = [col for col in features if col in df_rent.columns]
        if not valid_features:
            st.warning('No matching feature columns in df_rent')
            return
        
        hotel_features = df_rent[['name'] + valid_features].drop_duplicates(subset='name')
        liked_hotels = df_rent[(df_rent['user_id'] == user_id) & (df_rent['rating'] >= 4)]
        liked_hotels_names = liked_hotels['name'].unique()
        liked_features = hotel_features[hotel_features['name'].isin(liked_hotels_names)].set_index('name')

        if liked_features.empty:
            st.info('No hotels found that are liked by the user')
            return
        
        hotel_encoded = pd.get_dummies(hotel_features.set_index('name'))
        liked_encoded = pd.get_dummies(liked_features)
        liked_encoded = liked_encoded.reindex(columns=hotel_encoded.columns, fill_value=0)

        user_profile = liked_encoded.mean()
        scores = hotel_encoded.dot(user_profile)
        scores_percent = 100 * (scores - scores.min()) / (scores.max() - scores.min())

        rated_hotels = df_rent[df_rent['user_id'] == user_id]['name'].unique()

        recommendations = scores_percent.sort_values(ascending=False).head(top_n).reset_index()
        recommendations.columns = ['name','similarity_percent']

        for _, row in recommendations.iterrows():
            info = df_rent[df_rent['name'] == row['name']].iloc[0]
            with st.container():
                st.markdown(f'**Hotel: {row['name']}**')
                st.markdown(f'Location: {info['city']}, {info['detailed_address']}')
                st.markdown(f'Bedrooms: {info['bedrooms']} | Bathrooms: {info['bathrooms']} | Guests: {info['number_of_guests']}')
                st.markdown(f'Amenities: Jungle View: {info['Jungle View']} | Surfing: {info['Surfing']} | Pool View: {info['Pool view']} | Golfing: {info['Golfing']}')
                st.markdown(f'Price: {info['price_value']} {info['currency']}')
                # st.markdown(f" Similarity Score: {row['similarity_percent']:.2f}%")
                st.markdown("---")

# =======INPUT BUTTON ============
top_n = st.slider('Number of Recommendations', min_value=1, max_value=10, value=5)
alpha = st.slider('Weight for Models(0 = SVD, 1 = RF)', min_value=0.0, max_value=1.0, value=0.5)
if st.button('Generate recommendations'):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Hybrid Model')
        hybrid_recommendation(user_id=input_user_id, top_n=top_n, alpha=alpha)
    with col2:
        st.subheader('Similarity Based')
        show_content_based_recommendation(user_id=input_user_id, top_n=top_n)    



 
