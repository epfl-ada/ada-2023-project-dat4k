import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import scipy.stats 
import ast
import calendar
import networkx as nx
import re

##### MONTH PROCESSING HELPERS

def select_years(df):
    '''
    Select only the row with known movie release date, and create a new column 'Movie release year'.
    '''
    # Drop nan values
    df = df.dropna(subset=['Movie release date']).copy()  # Make a copy to avoid chained assignment
    # Add release year column
    df['Movie release year'] = df['Movie release date'].str[0:4]
    # Convert to numeric values
    df['Movie release year'] = pd.to_numeric(df['Movie release year'], errors='raise') 
    # Sort the movies by ascending order of release year
    df = df.sort_values('Movie release year', ascending=True) 
    # Drop the first row which has an error in the release year (1010)
    df = df.drop(df[df['Movie release year'] == 1010].index)
    # Reset the index
    df = df.reset_index(drop=True)

    return df

def dataframe_with_months (df):
    '''
    Return the dataframe with only the row for which the month of release is known.
    '''
    # Remove the row which don't have the month of release
    df = df[df['Movie release date'].str.len() > 4]
    # Reset the indexation 
    df = df.reset_index(drop = True)
    return df

'''def select_main_years(df1,df2):
    film_counts_year_without_missing_months = df2['Movie release year'].value_counts().sort_index()
    years_under_200 = film_counts_year_without_missing_months.index[film_counts_year_without_missing_months.values > 200]
    df1 = df1[df1['Movie release year'].isin(years_under_200)]
    return df1'''

def select_main_years(df1):
    '''
    Delete all rows for which the number of movies in the corresponding year is lower than 200.
    '''

    film_counts_year_without_missing_months = df1['Movie release year'].value_counts().sort_index()
    years_under_200 = film_counts_year_without_missing_months.index[film_counts_year_without_missing_months.values > 200]
    df1 = df1[df1['Movie release year'].isin(years_under_200)]
    return df1

def clean_date_and_season (df):
    '''
    Return a dataframe with 2 new columns: 'Movie release month', 'Movie release season'
    
    '''
    # Create a column with only the release month 
    df['Movie release month'] = df['Movie release date'].str[5:7]
    #Convert to numeric the release months
    df['Movie release month'] = pd.to_numeric(df['Movie release month'], errors='raise') 

    # Sort the movies by ascending order of release year
    df = df.sort_values('Movie release year', ascending=True) 

    # Remove the Movie release date column
    df = df.drop(columns=['Movie release date'])

    # Reset the indexation 
    df = df.reset_index(drop = True) 

    # Add the season column
    df['Movie release season'] = df['Movie release month'].apply(lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4)

    return df

##### VISUALIZE HELPERS

def nmbr_movie_years(df1,df2):
    '''
    Plot the number of movies per year for the unfiltered data (df1), and also the data for which the month of release is known (df2).
    '''

    # Counting number of movie per year
    film_counts_year = df1['Movie release year'].value_counts().sort_index()
    # Counting number of movie with month release per year
    film_counts_year_without_missing_months = df2['Movie release year'].value_counts().sort_index()

    # Plot the number of movies per year
    plt.subplot(1, 2, 1)

    plt.bar(film_counts_year.index, film_counts_year.values, color='orange')
    plt.title('Number of movie per year')
    plt.xlabel('Release year')
    plt.ylabel('Number of movie')
    plt.grid()

    # Plot the number of movies per year
    plt.subplot(1, 2, 2)

    plt.bar(film_counts_year_without_missing_months.index, film_counts_year_without_missing_months.values, color='orange')
    plt.title('Number of movie per year\nwith month release')
    plt.xlabel('Release year')
    plt.ylabel('Number of movie')
    plt.grid()

    # Set the same Y-axis limits for both subplots
    plt.ylim(min(min(film_counts_year.values), min(film_counts_year_without_missing_months.values)),
            max(max(film_counts_year.values), max(film_counts_year_without_missing_months.values)))

    # Adjust layout
    plt.tight_layout()
    # Show the plots
    plt.show()

def plot_percentage_missing_month_per_year(df):
    '''
    Plot the percentage of missing month data per year.
    '''
    percentage = lambda x: (x.astype(str).apply(len) < 5).mean() * 100

    missing_data_percentage = df.groupby('Movie release year')['Movie release date'].apply(percentage)

    plt.plot(missing_data_percentage.index, missing_data_percentage.values, marker='.')
    plt.title('Percentage of missing month per year')
    plt.xlabel('Year')
    plt.ylabel('Percentage missing [%]')
    plt.show()

def visualizing_data(df, split_year, genre):

    #reducing columns to see clearly
    df_genres = df[['Movie name','genre 1', 'genre 2', 'Movie release month', 'Movie release year']]
    #all years
    df_genres=df_genres[~(df_genres['genre 2'].isna() & df_genres['genre 1'].isna())] #removing when there 2 NaN : we lose around 4000 movies
    film_counts_month = df_genres['Movie release month'].value_counts().sort_index() #film by month for ratio
    #after split year
    df_after=df_genres[df_genres['Movie release year']>=split_year]  #After 1990 42k -> 20k
    film_counts_month_a = df_after['Movie release month'].value_counts().sort_index() #film by month for ratio
    #before split year
    df_before=df_genres[df_genres['Movie release year']<split_year]  #Before 1990 42k -> 22k
    film_counts_month_b = df_before['Movie release month'].value_counts().sort_index() #film by month for ratio
    
    #plt.subplot(2, 2, 1)
    genre_distribution_over_month(df_after, genre, film_counts_month_a, split_year, 'a')
    #plt.subplot(2, 2, 2)
    genre_distribution_over_month(df_before, genre, film_counts_month_b, split_year, 'b')
    #plt.subplot(2, 2, 3)
    genre_distribution_over_month(df_genres, genre, film_counts_month, split_year, 'c')
    plt.show


def genre_distribution_over_month(df, genre, film_counts_month, split_year, when):  #df_genres, #Drama, #film_counts_month
    df_selected_gender=df[(df['genre 1'] == genre ) | (df['genre 2'] == genre)] #selecting genre
    genre_distrib = df_selected_gender.groupby('Movie release month').count()['Movie name'] #genre distrib over months
    
    # Create the first subplot for the bar plot
    plt.figure(figsize=(4,3))
    plt.subplot(2, 1, 1)
    plt.bar(genre_distrib.index, genre_distrib.values, color='orange')
    if when == 'c':
        plt.title(f'Number of {genre} movie per month for all years')
    elif when == 'b':
        plt.title(f'Number of {genre} movie per month before {split_year}')
    elif when == 'a':
        plt.title(f'Number of {genre} movie per month after {split_year}')
    plt.ylabel(f'Number of {genre} movie')
    plt.grid()

    # Create the second subplot for the line plot
    plt.subplot(2, 1, 2)
    plt.bar(film_counts_month.index, genre_distrib.values/film_counts_month.values, color='blue')
    if when == 'c':
        plt.title(f'Ratio of {genre} movie per month for all years')
    elif when == 'b':
        plt.title(f'Ratio of {genre} movie per month before {split_year}')
    elif when == 'a':
        plt.title(f'Ratio of {genre} movie per month after {split_year}')
    plt.xlabel('Release month')
    plt.ylabel(f'ratio of {genre} movies ')
    plt.grid()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()   

def visu_P2(df, split_year, genre):
    #reducing colums to see clearly
    df_genres = df[['Movie name','genre 1', 'genre 2', 'Movie release month', 'Movie release year']]
    #after split year
    df_after=df_genres[df_genres['Movie release year']>=split_year]  #After 1990 42k -> 20k
    film_counts_month = df_after['Movie release month'].value_counts().sort_index() #film by month for ratio
    df_selected_gender2=df[(df['genre 1'] == genre ) | (df['genre 2'] == genre)] #selecting genre

    df_selected_gender=df_after[(df_after['genre 1'] == genre ) | (df_after['genre 2'] == genre)] #selecting genre
    genre_distrib = df_selected_gender.groupby('Movie release month').count()['Movie name'] #genre distrib over months
    
    
    genre_counts = df_selected_gender2.groupby('Movie release year').size() # Count number of movie of a specific genre in every year
    
     # Create the first subplot for the bar plot
    
    plt.figure(figsize=(6,6))
    plt.subplot(3,1,1)
    genre_counts.plot(kind='line', color='skyblue')
    plt.title(f'Number of {genre} movies in every year ')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
   

   
    plt.subplot(3, 1, 2)
    plt.bar(genre_distrib.index, genre_distrib.values, color='orange')
    plt.title(f'Number of {genre} movie per month after {split_year}')
    plt.ylabel(f'Number of {genre} movie')
    plt.grid()

    # Create the second subplot for the line plot
    plt.subplot(3, 1, 3)
    plt.bar(film_counts_month.index, genre_distrib.values/film_counts_month.values, color='blue')
    plt.title(f'Ratio of {genre} movie per month after {split_year}')
    plt.xlabel('Release month')
    plt.ylabel(f'ratio of {genre} movies ')
    plt.grid()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()

def plot_general(df): 
    
    season_mapping = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    df['season'] = df['Movie release season'].map(season_mapping)
    melted_df = pd.melt(df, id_vars=['season'], value_vars=['genre 1', 'genre 2'], value_name='Genre_unique')
    genre_season_counts = melted_df.groupby(['Genre_unique', 'season']).size().reset_index(name='Nombre de films')

    # Plot
    plt.figure(figsize=(12, 10))
    plt.subplot(2,1,1)
    sns.barplot(x='Genre_unique', y='Nombre de films', hue='season', data=genre_season_counts)
    plt.title('Number of movies of each season for different genres')
    plt.xlabel('Genre')
    plt.ylabel('Number of movies')

    plt.subplot(2,1,2)
    sns.barplot(x='season', y='Nombre de films', hue='Genre_unique', data=genre_season_counts)
    plt.title('Number of movies of each genre in every seasons')
    plt.xlabel('Season of release')
    plt.ylabel('Number of movies')
    plt.xticks(ticks=[0, 1, 2, 3], labels=['Winter', 'Spring', 'Summer', 'Fall'])
    plt.show()

def plot_average_monthly_revenue(df_clean_genre_boxoffice, percentage = False):

    #Declare a dataframe per main genre

    drama_genres = df_clean_genre_boxoffice[(df_clean_genre_boxoffice['genre 1']=='Drama')|(df_clean_genre_boxoffice['genre 2']=='Drama')]
    comedy_genres = df_clean_genre_boxoffice[(df_clean_genre_boxoffice['genre 1']=='Comedy')|(df_clean_genre_boxoffice['genre 2']=='Comedy')]
    romance_genres = df_clean_genre_boxoffice[(df_clean_genre_boxoffice['genre 1']=='Romance')|(df_clean_genre_boxoffice['genre 2']=='Romance')]
    thriller_genres = df_clean_genre_boxoffice[(df_clean_genre_boxoffice['genre 1']=='Thriller')|(df_clean_genre_boxoffice['genre 2']=='Thriller')]
    action_genres = df_clean_genre_boxoffice[(df_clean_genre_boxoffice['genre 1']=='Action')|(df_clean_genre_boxoffice['genre 2']=='Action')]
    family_genres = df_clean_genre_boxoffice[(df_clean_genre_boxoffice['genre 1']=='Family film')|(df_clean_genre_boxoffice['genre 2']=='Family film')]
    horror_genres = df_clean_genre_boxoffice[(df_clean_genre_boxoffice['genre 1']=='Horror')|(df_clean_genre_boxoffice['genre 2']=='Horror')]
    informative_genres = df_clean_genre_boxoffice[(df_clean_genre_boxoffice['genre 1']=='Informative')|(df_clean_genre_boxoffice['genre 2']=='Informative')]

    #Computations of the total revenue per month for each genre

    monthly_revenue_drama = drama_genres.groupby(drama_genres['Movie release month'])['Movie box office revenue'].sum()
    monthly_revenue_comedy = comedy_genres.groupby(comedy_genres['Movie release month'])['Movie box office revenue'].sum()
    monthly_revenue_romance = romance_genres.groupby(romance_genres['Movie release month'])['Movie box office revenue'].sum()
    monthly_revenue_thriller = thriller_genres.groupby(thriller_genres['Movie release month'])['Movie box office revenue'].sum()
    monthly_revenue_family = family_genres.groupby(family_genres['Movie release month'])['Movie box office revenue'].sum()
    monthly_revenue_action = action_genres.groupby(action_genres['Movie release month'])['Movie box office revenue'].sum()
    monthly_revenue_horror = horror_genres.groupby(horror_genres['Movie release month'])['Movie box office revenue'].sum()
    monthly_revenue_informative = informative_genres.groupby(informative_genres['Movie release month'])['Movie box office revenue'].sum()

    if percentage == False :
        # calculate the average revenue per month
        average_monthly_revenue_drama = monthly_revenue_drama.groupby(level=0).mean()
        average_monthly_revenue_comedy = monthly_revenue_comedy.groupby(level=0).mean()
        average_monthly_revenue_romance = monthly_revenue_romance.groupby(level=0).mean()
        average_monthly_revenue_thriller = monthly_revenue_thriller.groupby(level=0).mean()
        average_monthly_revenue_family = monthly_revenue_family.groupby(level=0).mean()
        average_monthly_revenue_action = monthly_revenue_action.groupby(level=0).mean()
        average_monthly_revenue_horror = monthly_revenue_horror.groupby(level=0).mean()
        average_monthly_revenue_informative = monthly_revenue_informative.groupby(level=0).mean()

    else:
        
        total = monthly_revenue_drama + monthly_revenue_comedy + monthly_revenue_romance + monthly_revenue_thriller + monthly_revenue_action + monthly_revenue_family

        # Calculate the average revenue per month per movie
        average_monthly_revenue_drama = monthly_revenue_drama.groupby(level=0).mean() / total*100
        average_monthly_revenue_comedy = monthly_revenue_comedy.groupby(level=0).mean() / total*100
        average_monthly_revenue_romance = monthly_revenue_romance.groupby(level=0).mean() / total*100
        average_monthly_revenue_thriller = monthly_revenue_thriller.groupby(level=0).mean() / total*100
        average_monthly_revenue_action = monthly_revenue_action.groupby(level=0).mean() / total*100
        average_monthly_revenue_family = monthly_revenue_family.groupby(level=0).mean() / total*100
        average_monthly_revenue_horror = monthly_revenue_horror.groupby(level=0).mean() / total*100
        average_monthly_revenue_informative = monthly_revenue_informative.groupby(level=0).mean() / total*100


    average_monthly_revenue_drama.plot(kind='line', title='Average Box Office Revenue per Month for Different Genres', label='Drama')
    average_monthly_revenue_comedy.plot(kind='line', label='Comedy')
    average_monthly_revenue_romance.plot(kind='line', label='Romance')
    average_monthly_revenue_thriller.plot(kind='line', label='Thriller')
    average_monthly_revenue_family.plot(kind='line', label='Family')
    average_monthly_revenue_action.plot(kind='line', label='Action')
    average_monthly_revenue_horror.plot(kind='line', label='Horror')
    average_monthly_revenue_informative.plot(kind='line', label='Informative')

    plt.xlabel('Month')
    plt.ylabel('Average Box Office Revenue')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.xlim(1, 12)
    plt.legend()
    plt.show()

def nb_movies_genres(nb_genres):
    plt.scatter(nb_genres.index,nb_genres['nb of movies'],facecolors='none', edgecolors='b')
    plt.yscale("log")
    plt.title('Number of movies per genre')
    plt.xlabel('Index of the genres in ascending order of occurrence')
    plt.ylabel('Number of movie in log scale')
    plt.grid()

##### GENRES HELPERS 

#Gathering all genres and their occurrences
def counting_genres(df):
    pd.options.mode.chained_assignment = None  # default='warn'

    # Create an empty dataframe
    # nb_genres := each genre names and their respective occurrence
    empty_frame = pd.DataFrame(index=range(363),columns=range(2))
    nb_genres = empty_frame.rename(columns={0: 'genre name', 1: 'nb of movies'})
    nb_genres['nb of movies'].fillna(0,inplace=True)
    i = 0

    # Iterate over rows of df
    for index, row in df.iterrows():
        df2 = row['Movie genres']
        df3 = json.loads(df2)
        df4 = pd.json_normalize(df3)
    
        for column in df4:
            if not (nb_genres['genre name'].isin([df4[column].iloc[0]]).any()):
                nb_genres['genre name'].iloc[i] = df4[column].iloc[0]
                nb_genres['nb of movies'][i] = 1
                i = i+1
            else:
                idx = nb_genres.loc[nb_genres['genre name'].isin([df4[column].iloc[0]])].index
                nb_genres['nb of movies'][idx] = nb_genres['nb of movies'][idx] + 1

    # Sort the values in ascending order of the number of movies
    nb_genres = nb_genres.sort_values("nb of movies",ascending=False)

    # Reset indexation
    nb_genres = nb_genres.reset_index(drop = True) 

    # Drop nan values
    nb_genres=nb_genres.dropna()

    return nb_genres

# Creating main genre cluster

def main_genres_cluster(genres_lexical_field,nb_genres):

    # Create a dataframe out of the dict input 
    genres_lexical_field = dict([ (k,pd.Series(v)) for k,v in genres_lexical_field.items() ])
    genres_lexical_field = pd.DataFrame(genres_lexical_field)
    genres_lexical_field

    # Create an empty dataframe
    main_genres = pd.DataFrame(columns = ['genre name', 'nb of movies', 'main name'])

    # Filling 'main_genres' by iterating over 'genres_lexical_field' to select the sub-genres that belong to specific lexical fields
    for column in genres_lexical_field:
        for index, row in genres_lexical_field.iterrows():
            if (not pd.isnull(genres_lexical_field[column].loc[index])):
                df_temp = nb_genres[nb_genres['genre name'].str.contains(genres_lexical_field[column].loc[index])]
                df_temp['main name'] = column
                main_genres = pd.concat([main_genres,df_temp])

    main_genres = main_genres.reset_index(drop = True)

    # Cases where the cluster went wrong: if some sub-genres are associated to an unrelated main genre
    # if a sub-genre related to Drama is associated to the main genre genre film
    if ((main_genres['main name'] == "genre film") & (main_genres['genre name'].str.contains('Drama'))).any():
        main_genres = main_genres.drop(main_genres[(main_genres['main name'] == "genre film") & (main_genres['genre name'].str.contains('Drama'))].index)
        main_genres = main_genres.reset_index(drop = True)

    # if a sub-genre related to Comedy is associated to the main genre Thriller
    if ((main_genres['main name'] == "Thriller") & (main_genres['genre name'].str.contains('Comedy'))).any():
        main_genres = main_genres.drop(main_genres[(main_genres['main name'] == "Thriller") & (main_genres['genre name'].str.contains('Comedy'))].index)
        main_genres = main_genres.reset_index(drop = True)

    return main_genres

#  Assigning up to 2 main genres to each movie

def reshape_genre_column(df,main_genres):
    # Deep copy
    df_clean_genre = df.copy(deep=True)
    # Create empty column for 2 main genres
    df_clean_genre['genre 1']= None
    df_clean_genre['genre 2'] = None
    amm = True
    # Iterate over rows of df_clean_genre
    for index, row in df_clean_genre.iterrows():
            df_clean_genre2 = df_clean_genre.iloc[index]['Movie genres']
            df_clean_genre3 = json.loads(df_clean_genre2)
            df_clean_genre4 = pd.json_normalize(df_clean_genre3)

            for column in df_clean_genre4:
                    boolarr = (main_genres['genre name'].isin([df_clean_genre4[column].iloc[0]]))
                    
                    
                    if (boolarr.sum() ==1):
                            main_genre_value = main_genres[boolarr]['main name'].iloc[0]
                            if (df_clean_genre['genre 1'].iloc[index] == None):
                                    df_clean_genre['genre 1'].iloc[index] = main_genre_value
                            elif (df_clean_genre['genre 2'].iloc[index] == None and df_clean_genre['genre 1'].iloc[index]!= main_genre_value):
                                    df_clean_genre['genre 2'].iloc[index] = main_genre_value
                    #case where a sub genre belongs to more than one main genre: ex: 'Crime Comedy' (iloc[796])
                    if (boolarr.sum() ==2):
                            if (df_clean_genre['genre 1'].iloc[index] == None and df_clean_genre['genre 2'].iloc[index] == None):
                                    df_clean_genre['genre 1'].iloc[index] = main_genres[boolarr]['main name'].iloc[0]
                                    df_clean_genre['genre 2'].iloc[index] = main_genres[boolarr]['main name'].iloc[1]
            
    # Drop the 'Movie genres' column
    df_clean_genre = df_clean_genre.drop(columns=['Movie genres'])

    return df_clean_genre

#Link between horror movies and month of release

def genre_distribution_over_month(df, genre, film_counts_month):  #df_genres, #Drama, #film_counts_month
    df_selected_gender=df[(df['genre 1'] == genre ) | (df['genre 2'] == genre)] #selecting genre
    genre_distrib = df_selected_gender.groupby('Movie release month').count()['Movie name'] #genre distrib over months
    
    # Create the first subplot for the bar plot
    plt.subplot(2, 1, 1)
    plt.bar(genre_distrib.index, genre_distrib.values, color='orange')
    plt.title(f'Number of {genre} movie per Month for all Years')
    plt.ylabel(f'Number of {genre} movie')
    plt.grid()

    # Create the second subplot for the line plot
    plt.subplot(2, 1, 2)
    plt.bar(film_counts_month.index, genre_distrib.values/film_counts_month.values, color='blue')
    plt.title(f'ratio of {genre} movie over film count by month')
    plt.xlabel('Release Month')
    plt.ylabel(f'ratio of {genre} movies ')
    plt.grid()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()

##### CONTINENT HELPERS

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

##### T-TEST HELPER

def ttest(df,genre, months, year_split):
    months = pd.Series(months)
    months_size = months.shape[0]
    not_months_size = 12-months.shape[0]
    months_names = months.apply(lambda x: calendar.month_abbr[x])
    if year_split == None:
        nb_genre_in_month = df[(df['Movie release month'].isin(months)) & ((df['genre 1']==genre) | 
                                                        (df['genre 2']== genre))] 
        nb_genre_in_month_by_year = nb_genre_in_month.groupby(['Movie release year'])['Freebase movieID'].count()
        print(f'The average number of',genre,'movies in',months_names.values,f'by year is: {(nb_genre_in_month_by_year/months_size).mean():.2f}') #calendar.month_name[months]
        
        nb_genre_not_in_month = df[(~ df['Movie release month'].isin(months)) & ((df['genre 1']==genre) | 
                                                        (df['genre 2']== genre))] 
        nb_genre_not_in_month_by_year = nb_genre_not_in_month.groupby(['Movie release year'])['Freebase movieID'].count()
        print(f'The average number of',genre,'movies not in',months_names.values,f'by year is: {(nb_genre_not_in_month_by_year/not_months_size).mean():.2f}') 
    
    else:
        nb_genre_in_month = df[(df['Movie release month'].isin(months)) & (df['Movie release year'] >= year_split)& ((df['genre 1']==genre) | 
                                                        (df['genre 2']== genre))] 
        nb_genre_in_month_by_year = nb_genre_in_month.groupby(['Movie release year'])['Freebase movieID'].count()
        print('The average number of',genre,'movies in',months_names.values,'by year (after the year',year_split,f') is: {(nb_genre_in_month_by_year/months_size).mean():.2f}')
        
        nb_genre_not_in_month = df[(~ df['Movie release month'].isin(months)) & (df['Movie release year'] >= year_split) & ((df['genre 1']==genre) | 
                                                        (df['genre 2']== genre))] 
        nb_genre_not_in_month_by_year = nb_genre_not_in_month.groupby(['Movie release year'])['Freebase movieID'].count()
        
        print('The average number of',genre,'movies not in',months_names.values,'by year (after the year',year_split,f') is: {(nb_genre_not_in_month_by_year/not_months_size).mean():.2f}') 

    ttest = scipy.stats.ttest_ind(np.round(nb_genre_in_month_by_year/months_size), np.round(nb_genre_not_in_month_by_year/not_months_size))
    if (ttest.pvalue < 0.05):
        print(f'The p-value for the t-test is equal to {ttest.pvalue:.6f} so the null hypothesis is rejected.')
    else:
        print(f'The p-value for the t-test is equal to {ttest.pvalue:.6f} so the null hypothesis is not rejected.')

    plt.plot((nb_genre_in_month_by_year/months_size),label=f"average nb of {genre} movies in {months_names.values}")
    plt.plot(np.round(nb_genre_not_in_month_by_year/not_months_size),label=f"average nb {genre} movies not in {months_names.values}")
    plt.legend()
    plt.title(f"Number of {genre} movies per year")

##### PAIRED MATCHING HELPERS
    
def print_mean_std(treatment, control, column):
    treatment_column_mean = treatment[column].mean()
    control_column_mean = control[column].mean()

    treatment_column_std = treatment[column].std()
    control_column_std = control[column].std()

    print('The mean',column,'for the treatment group is',round(treatment_column_mean,2),', and its std is',round(treatment_column_std,2))
    print('The mean',column,'for the control group is',round(control_column_mean,2),', and its std is',round(control_column_std,2))

def get_similarity(propensity_score1, propensity_score2):
    '''Calculate similarity for instances with given propensity scores'''
    return 1-np.abs(propensity_score1-propensity_score2)

def paired_matching(df):

    # Separate the treatment and control groups
    treatment_df = df[df['treat'] == 1]
    control_df = df[df['treat'] == 0]

    # Create an empty undirected graph
    G = nx.Graph()

    # Loop through all the pairs of instances
    for control_id, control_row in control_df.iterrows():
        for treatment_id, treatment_row in treatment_df.iterrows():

            # Force equality for the feature 'Northern_America'
            if (control_row['Northern_America'] == treatment_row['Northern_America']):
                # Calculate the similarity 
                similarity = get_similarity(control_row['Propensity_score'],
                                            treatment_row['Propensity_score'])

                # Add an edge between the two instances weighted by the similarity between them
                G.add_weighted_edges_from([(control_id, treatment_id, similarity)])

    # Generate and return the maximum weight matching on the generated graph
    matching = nx.max_weight_matching(G)

    matched = [i[0] for i in list(matching)] + [i[1] for i in list(matching)]

    balanced_df = df.iloc[matched]

    balanced_treatment = balanced_df.loc[balanced_df['treat'] == 1] 
    balanced_control = balanced_df.loc[balanced_df['treat'] == 0] 
    
    return balanced_df, balanced_treatment, balanced_control

### MACHINE LEARNING ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import seaborn as sns
import json 
import ast
import random

from scipy import stats
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import display
from nltk.corpus import wordnet

#from nltk.corpus import wordnet
from itertools import chain

#Vader
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#Scikit imports
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')

nltk.download('stopwords')
nltk.download('punkt')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 50)

# Load WordNet synsets for each season-related category
summer_synsets = wordnet.synsets('summer')
winter_synsets = wordnet.synsets('winter')
fall_synsets = wordnet.synsets('fall')
spring_synsets = wordnet.synsets('spring')
# Extract words from synsets
summer_words = set([lemma.name() for synset in summer_synsets for lemma in synset.lemmas()])
winter_words = set([lemma.name() for synset in winter_synsets for lemma in synset.lemmas()])
fall_words = set([lemma.name() for synset in fall_synsets for lemma in synset.lemmas()])
spring_words = set([lemma.name() for synset in spring_synsets for lemma in synset.lemmas()])
# Combine all seasonal words into one list
all_seasonal_words = list(summer_words.union(winter_words, fall_words))

# Filter the list to include only words found in WordNet
valid_seasonal_words = [word for word in all_seasonal_words if wordnet.synsets(word)]

# Print the valid seasonal words
#print("Valid Seasonal Words:", valid_seasonal_words)

def boxplots(genre_movies, treatment, control):

     # Create three subplots side by side
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    df = genre_movies[['Movie box office revenue','budget', 'Movie release year']]
    # Plot each boxplot on a separate subplot
    for i, column in enumerate(df.columns[:2]):
        #df.boxplot(column=column, ax=axes[i], grid=True)
        genre_movies.boxplot(by='treat', column=column, ax=axes[0, i], figsize = [5, 5], grid=True)
        axes[0, i].set_title(f'Boxplot for {column}', fontsize= 16)
        axes[0, i].set_xlabel('treat', fontsize= 14)
        #axes[i].set_ylabel('Values')
    # Plot a single boxplot for the third group below
    genre_movies.boxplot(by='treat', column='Movie release year', ax=axes[1, 0], grid=True)
    axes[1, 0].set_title(f'Boxplot for Movie release year', fontsize=16)
    axes[1, 0].set_xlabel('treat', fontsize=14)

    # Hide the empty subplot in the second row and second column
    axes[1, 1].axis('off')
    #Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

    print_mean_std(treatment, control, 'Movie box office revenue')
    print('\n')
    print_mean_std(treatment, control, 'budget')
    print('\n')
    print_mean_std(treatment, control, 'Movie release year')
    print('\n')

def lexical():
     # Spring-related words
    spring_lexicon = [
        'blossom', 'bloom', 'thaw', 'rain', 'sprout', 'garden', 'sunshine',
        'renewal', 'fresh', 'growth', 'bud', 'warmth', 'green', 'nature',
        'cherry', 'lilac', 'daffodil', 'tulip', 'pastel', 'bees', 'butterfly',
        'songbird', 'new','life', 'sunbeam', 'gentle','breeze', 'blossom', 'spring','reborn'
    ]

    # Summer-related words
    summer_lexicon = [
        'sunshine', 'heat', 'beach', 'vacation', 'barbecue', 'swim', 'sunny',
        'warm', 'leisure', 'picnic', 'relaxation', 'holiday', 'tan',
        'surfing', 'sandcastle', 'palm', 'ice_cream', 'flip_flops', 'sunglasses','summer',
        'heatwave', 'fireworks', 'tropical', 'sundress', 'sunset', 'poolside', 'independence'
    ]

    # Autumn (Fall)-related words
    autumn_lexicon = [
        'foliage', 'harvest', 'cool', 'pumpkin', 'cider', 'crisp', 'colorful',
        'sweater', 'apple', 'acorn', 'cozy', 'spice', 'hayride', 'autumnal',
        'cornucopia', 'cranberry', 'cornmaze', 'fireside', 'windy','atumn',
        'hoodie', 'mushroom', 'chestnut', 'sweater','weather', 'mild_temperature', 'plaid','orange','leaves','halloween'
    ]

        # Winter-related words
    winter_lexicon = [
        'snow', 'frost', 'ice', 'holiday', 'fireplace', 'cozy', 'festive', 'chill', 'Hanukkah',
        'mitten', 'scarf', 'celebrate', 'snowflake', 'hibernate', 'wonderland',
        'icicle', 'penguin', 'mittens', 'cocoa', 'blanket','ski','chairlift','christmas','santa','claus','romantic','valentine','roses','sex'
        'evergreen', 'frosty_morning', 'winter','solstice', 'snowstorm', 'jingle', 'toasty', 'snowy'
    ]

    # Valentine's Day related words
    valentines_day_lexicon = ["valentine", "romance", "cupid",  "flowers"]

    # Christmas related words
    christmas_lexicon = ["christmas", "holiday", "santa", "gift", "tree", "ornament", "jingle", "bell", "sleigh", "merry"]

    # New Year's related words
    new_years_lexicon = ["new year", "celebrate", "countdown", "party", "fireworks", "resolution"]

    halloween_lexicon = ["halloween", "spooky", "ghost", "witch", "pumpkin", "candy", "costume", "trick", "haunt", "skeleton","fear","scared","horror"]

    autumn_celebrations_lexicon = ["autumn", "Thanksgiving", "Halloween", "harvest", "turkey", "pilgrims"]

    spring_celebrations_lexicon = ["spring", "Easter", "Passover", "blossom", "renewal", "bunny", "egg","chocolate"]

    # Combine all lexicons
    winter_lexicon = winter_lexicon + valentines_day_lexicon + christmas_lexicon + new_years_lexicon

    autumn_lexicon = autumn_lexicon + autumn_celebrations_lexicon+halloween_lexicon

    spring_lexicon = spring_lexicon + spring_celebrations_lexicon

    # Combine all lexicons
    season_lexicon = spring_lexicon + summer_lexicon + autumn_lexicon + winter_lexicon



    # Function to get synonyms from WordNet
    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    # Expand the season-related lexicons with synonyms
    expanded_spring_lexicon = list(chain.from_iterable([spring_word] + get_synonyms(spring_word) for spring_word in spring_lexicon))
    expanded_summer_lexicon = list(chain.from_iterable([summer_word] + get_synonyms(summer_word) for summer_word in summer_lexicon))
    expanded_autumn_lexicon = list(chain.from_iterable([autumn_word] + get_synonyms(autumn_word) for autumn_word in autumn_lexicon))
    expanded_winter_lexicon = list(chain.from_iterable([winter_word] + get_synonyms(winter_word) for winter_word in winter_lexicon))

    # Combine all expanded lexicons
    expanded_season_lexicon = expanded_spring_lexicon + expanded_summer_lexicon + expanded_autumn_lexicon + expanded_winter_lexicon

    return expanded_season_lexicon


def text_analisis(lines_plots, lexique, nlp, df):
     
    #tokenized_summaries = {}
    movie_ids = []
    processed_corpus = []

    for line in lines_plots:
        # Split each line into movie ID and summary
        parts = re.split('[\t ]', line, 1)
        if len(parts) == 2:
            movie_id, summary = parts
            movie_id = int(movie_id)
            if movie_id in df['Wikipedia movieID'].values:  
                #stemmed_words = [porter.stem(word) for word in words] 
                doc = nlp(summary)
                processed_tokens=[]
                for token in doc:               
                    if token.lemma_ in lexique:
                        processed_tokens.extend([token.lemma_])  
                if len(processed_tokens)>0:
                    movie_ids.append(movie_id)
                    processed_summary = ' '.join(processed_tokens)# for token in doc if token in expanded_season_lexicon])# and token.pos_ != '-PRON-'])
                    processed_corpus.append(processed_summary)

    vectorizer = CountVectorizer()

    bow_matrix = vectorizer.fit_transform(processed_corpus)
    bow_array = bow_matrix.toarray()
    matrix = bow_array

    # Normalize each row by the number of elements
    matrix = np.divide(bow_array, bow_array.sum(axis=1, keepdims=True))

    #As the loop takes a lot of time, we save the obtained matrix as a df.csv
    MAT = pd.DataFrame(matrix, columns=vectorizer.get_feature_names_out())
    MAT.insert(0, 'Movie ID', movie_ids)
    MAT.to_csv('Data/BOW.csv', index=False)

    return MAT

def sentiment_analisis (df_sentiment,lines_plots,nlp, df_all):

    for line in lines_plots:
        parts = re.split('[\t ]', line, 1)
        if len(parts) == 2:

            #Separation of the movie ID and its plot summary
            movie_id, summary = parts
            movie_id = int(movie_id)

            # Check if the movie is present in movie.metadata.genres.tsv
            if movie_id in df_all['Wikipedia movieID'].values:
                
                #NLP processing
                doc = nlp(summary)

                #Calculus of sentiments coefficients
                analyzer = SentimentIntensityAnalyzer()
                vs = analyzer.polarity_scores(doc.text)
                
                #Adding complete rows to df_sentiment
                new_line = ({'Movie_ID' : movie_id, 'Positive' : vs['pos'], 'Negative' : vs['neg'], 'Compound' : vs['compound'], 'Neutral' : vs['neu']})
                df_sentiment = pd.concat([df_sentiment, pd.DataFrame([new_line])], ignore_index=True)     

    #As the loop takes a long time to run we save the dataframe obtain into a csv file
    file_path = 'Data\df_sentiment.csv'
    df_sentiment.to_csv(file_path, index=False)

    return df_sentiment


def random_ttest(predictions, test):

    model_scores=[]
    random_classifier_scores=[]

    for i in range(len(predictions)):   
        if predictions[i] == test[i]:
            model_scores.append(1)
        else:
            model_scores.append(0)
        ran = random.randint(1,4)
        if ran == test[i]:
            random_classifier_scores.append(1)
        else:
            random_classifier_scores.append(0)
    
    t_statistic, p_value = stats.ttest_ind(random_classifier_scores, model_scores)

    return t_statistic, p_value, random_classifier_scores, model_scores
     
# Define a function for hyperparameter tuning and plotting
def tune_and_plot(model, param_grid, model_name, param_name, X, Y):

    kf = 5
    
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy')
    grid_search.fit(X, Y)
    
    # Extract the relevant information from the results
    param_values = param_grid[param_name]
    mean_scores = grid_search.cv_results_['mean_test_score']
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, mean_scores, marker='o')
    plt.title(f'{model_name} - {param_name} Tuning')
    plt.xlabel(f'{param_name}')
    plt.ylabel('Mean Accuracy')
    plt.grid(True)
    plt.show()

def what_would_be_significant(random_classifier_scores, model_scores):
    accs= [0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.35]
    alpha = 0.05

    num_samples = int(23700*0.2)

    for acc in accs:
        
        model_scores = np.random.binomial(1, acc, num_samples)
        t_statistic, p_value = stats.ttest_ind(random_classifier_scores, model_scores)
        print(f"Accuracy: {acc} p-value: {p_value} significant: {p_value < alpha}")

