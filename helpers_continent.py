import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 
import ast
from helpers import visualizing_data
import networkx as nx



def extract_nb_countries(df):
    # remove a warning
    pd.options.mode.chained_assignment = None  # default='warn'

    # Create an empty dataframe
    empty_frame = pd.DataFrame(index=range(363),columns=range(2))
    nb_countries = empty_frame.rename(columns={0: 'country', 1: 'nb of movies'})
    nb_countries['nb of movies'].fillna(0,inplace=True)
    i = 0

    dfc5 = None
    # Iterate over rows of df
    for index, row in df.iterrows():
        dfc2 = df.iloc[index]['Movie countries']
        dfc3 = json.loads(dfc2)
        dfc4 = pd.json_normalize(dfc3)
    
        for column in dfc4:
            if not (nb_countries['country'].isin([dfc4[column].iloc[0]]).any()):
                nb_countries['country'].iloc[i] = dfc4[column].iloc[0]
                nb_countries['nb of movies'][i] = 1
                i = i+1
            else:
                idx = nb_countries.loc[nb_countries['country'].isin([dfc4[column].iloc[0]])].index
                nb_countries['nb of movies'][idx] = nb_countries['nb of movies'][idx] + 1

    return nb_countries

def obtain_continents(nb_countries): 
    nb_countries = nb_countries.sort_values("nb of movies",ascending=False)
    nb_countries=nb_countries[(nb_countries['nb of movies']>=5)] #remove countries with less than 5 movies
    c = (nb_countries['nb of movies']>=5).sum()
    print(f"We don't classify countries with less than 5 movies which represents {c} movies") 
    nb_countries = nb_countries.reset_index(drop = True) #don't compute this over and over!!!!

    #Assign each country to its continent. Some exceptions : Australia in North America, culturally closer.
    #EUROPE
    searchfor_europe = ['France', 'Italy', 'United Kingdom', 'Slovak Republic', 'Russia', 'Germany', 'Spain', 'Netherlands', 'Sweden', 'Denmark', \
                    'Belgium', 'Ireland', 'Norway', 'Czech Republic', 'Finland', 'Switzerland', 'Portugal', 'Poland', 'Austria', \
                        'Hungary', 'England', 'Luxembourg', 'Romania', 'Iceland', 'Croatia', 'Greece', 'Serbia', 'Bulgaria', 'Slovakia', \
                            'Slovenia', 'Scotland', 'Estonia', 'Bosnia and Herzegovina', 'Lithuania', 'Soviet Union', 'Ukraine', 'Yugoslavia'\
                                'Czechoslovakia	', 'Albania	','Kingdom of Great Britain	', 'Serbia and Montenegro' ]
    Europe = nb_countries[nb_countries['country'].str.contains('|'.join(searchfor_europe))]
    Europe['continent'] = 'Europe'

    #NORTH AMERICA
    searchfor_northa = ['United States', 'Canada' , 'Mexico', 'Australia']
    northa = nb_countries[nb_countries['country'].str.contains('|'.join(searchfor_northa))]
    northa['continent'] = 'northa'

    #SOUTH AMERICA
    searchfor_southa = ['Brazil', 'Colombia' , 'Peru', 'Cuba', 'Puerto Rico', 'Venezuela', 'Uruguay', 'Jamaica', 'Argentina']
    southa = nb_countries[nb_countries['country'].str.contains('|'.join(searchfor_southa))]
    southa['continent'] = 'southa'

    #ASIA
    searchfor_Asia = ['China', 'Russia','Japan' , 'Nepal', 'South Korea', 'Singapore', 'Cambodia', 'Bangladesh', 'Vietnam', 'Lebanon', 'Burma', 'Sri Lanka',\
                   'Palestinian territories', 'Israel', 'Iraq', 'Republic of Macedonia', 'Korea', 'India', 'Hong Kong', 'Philippines', 'Turkey',\
                      'New Zealand', 'Thailand', 'Indonesia', 'Pakistan', 'Iran', 'Taiwan', 'Malaysia', 'United Arab Emirates', 'Afghanistan']
    Asia = nb_countries[nb_countries['country'].str.contains('|'.join(searchfor_Asia))]
    Asia['continent'] = 'Asia'

    #AFRICA
    searchfor_Africa = ['South Africa', 'Egypt', 'Morocco', 'Algeria', 'Kenya', 'Tunisia', 'Burkina Faso', 'Mali', 'Senegal', 'Democratic Republic of the Congo']
    Africa = nb_countries[nb_countries['country'].str.contains('|'.join(searchfor_Africa))]
    Africa['continent'] = 'Africa'

    #Creating the main genre dataframe so we can modify the original frame
    Continent =  pd.concat([Europe, northa, southa, Asia, Africa])
    Continent = Continent.reset_index(drop = True)

    return Continent

def continent_in_df(df, Continent):
    # Create empty column for 2 continents
    df['continent_1']= None
    df['continent_2'] = None


    # Iterate over rows of df
    for index, row in df.iterrows():
            df2 = df.iloc[index]['Movie countries']
            df3 = json.loads(df2)
            df4 = pd.json_normalize(df3)

            for column in df4:
                    boolarr = (Continent['country'].isin([df4[column].iloc[0]])) #array with of length len(country) with True for countries detected
                    if (boolarr.sum() ==1):
                            continent_detected = Continent[boolarr]['continent'].values 
                            #if a country has 2 countries with different continents, assign them to 2 colums. ex : courage mountain : USA, France
                            if (df['continent_1'].iloc[index] == None):
                                df['continent_1'].iloc[index] = continent_detected
                            elif (df['continent_2'].iloc[index] == None and df['continent_1'].iloc[index]!= continent_detected):
                                df['continent_2'].iloc[index] = continent_detected

    return df

def continent_to_digit(df):
     '''
    northa -> 1
    Europe -> 2
    southa -> 3
    Asia -> 4
    Africa -> 5
    '''
     
     a = df["continent_2"].count()
     print(f" Only {a} movies have 2 continents so we take only continent 1 into consideration ") 

     df=df.drop(columns='continent_2')
     df['continent_1'] = df['continent_1'].replace(['northa', 'Europe', 'southa', 'Asia', 'Africa'], [1, 2, 3, 4, 5])
     print("northa -> 1\nEurope -> 2\nsoutha -> 3\nAsia -> 4\nAfrica -> 5")
     return df

def adding_continents(df):
     nb_countries = extract_nb_countries(df)
     Continent = obtain_continents(nb_countries)
     df2 = continent_in_df(df, Continent)
     df3 = continent_to_digit(df2)
     return df3
     