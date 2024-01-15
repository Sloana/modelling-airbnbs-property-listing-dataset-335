import pandas as pd
import re
df = pd.read_csv(r"C:/Users/laura/OneDrive/Desktop/Data Science/airbnb-property-listings/tabular_data/listing.csv")
# create function that clean the data
def clean_tabular_data():
# remove rows that have issing values in *_rating attributes
    def remove_rows_with_missing_ratings(df):
        df.dropna(subset=['Cleanliness_rating','Accuracy_rating', 'Communication_rating', 'Location_rating','Check-in_rating', 'Value_rating'], inplace=True)
        return df
    # fill the missing values of beds, guests, bathrooms and bedrooms with 1
# in discription combine the list items into the same string and remove empty quotes, and  "About this space"prefix. 
    def combine_description_strings(df=remove_rows_with_missing_ratings(df)):
        df=pd.DataFrame(df)
        df_disc=df['Description']
        df_list=[]
        for i in range (0,889):
            df_disc.iloc[i]=str(df_disc.iloc[i]).replace('About this space','')
            df_disc.iloc[i]=str(df_disc.iloc[i]).replace(',', '')
            df_disc.iloc[i]=str(df_disc.iloc[i]).replace('"', '')
            df_disc.iloc[i]=str(df_disc.iloc[i]).replace('  ', '')
            df_disc.iloc[i]=str(df_disc.iloc[i]).replace("''", '')
            df_disc.iloc[i]=str(df_disc.iloc[i]).replace("'", '')
            df_disc.iloc[i]=str(df_disc.iloc[i]).replace("[", '')
            df_disc.iloc[i]=str(df_disc.iloc[i]).replace("]", '')
            df_list.append(df_disc.iloc[i])
        df['Description']=pd.DataFrame(df_list)
        # print(df)
        return df
     # fill the missing values of beds, guests, bathrooms and bedrooms with 1
    def set_default_feature_values(df=combine_description_strings(df=remove_rows_with_missing_ratings(df))):
        df['guests'] = df['guests'].fillna(1)
       
        df['beds'] = df['beds'].fillna(1)
        df['guests'] = df['guests'].fillna(1)
        df['bathrooms'] = df['bathrooms'].fillna(1)
        df['bedrooms'] = df['bedrooms'].fillna(1)
        return df
    return set_default_feature_values(df=combine_description_strings(df=remove_rows_with_missing_ratings(df)))
# call the function 
if __name__ == "__main__":
    clean_tabular_data()
    clean_tabular_data().to_csv("C:/Users/laura/OneDrive/Desktop/Data Science/airbnb-property-listings/tabular_data/clean_tabular_data.csv")

# extract the label: Price_Night and the numerical features
def load_airbnd(df=clean_tabular_data()):
    features = df.loc[:, ['guests', 'beds','bathrooms','Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','amenities_count']]
    lables = df.loc[:, df.columns == 'Price_Night']
    return (features, lables)
