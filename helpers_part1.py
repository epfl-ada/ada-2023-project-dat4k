import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 
import ast
from datetime import datetime

### Part 2.1) Link between horror movies and moth of release

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

### Part 1.3) Processing the genre column

# 1.3.1) Gathering all genres and their occurrences
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

# 1.3.2) Creating main genre cluster

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
    # if a sub-genre related to Drama is associated to the main genre Family film
    if ((main_genres['main name'] == "Family film") & (main_genres['genre name'].str.contains('Drama'))).any():
        main_genres = main_genres.drop(main_genres[(main_genres['main name'] == "Family film") & (main_genres['genre name'].str.contains('Drama'))].index)
        main_genres = main_genres.reset_index(drop = True)

    # if a sub-genre related to Comedy is associated to the main genre Thriller
    if ((main_genres['main name'] == "Thriller") & (main_genres['genre name'].str.contains('Comedy'))).any():
        main_genres = main_genres.drop(main_genres[(main_genres['main name'] == "Thriller") & (main_genres['genre name'].str.contains('Comedy'))].index)
        main_genres = main_genres.reset_index(drop = True)

    return main_genres

# 1.3.3) Assigning up to 2 main genres to each movie

def reshape_genre_column(df,main_genres):
    # Deep copy
    df_clean_genre = df.copy(deep=True)
    # Create empty column for 2 main genres
    df_clean_genre['genre 1']= None
    df_clean_genre['genre 2'] = None

    # Iterate over rows of df_clean_genre
    for index, row in df_clean_genre.iterrows():
            df_clean_genre2 = df_clean_genre.iloc[index]['Movie genres']
            df_clean_genre3 = json.loads(df_clean_genre2)
            df_clean_genre4 = pd.json_normalize(df_clean_genre3)

            for column in df_clean_genre4:
                    boolarr = (main_genres['genre name'].isin([df_clean_genre4[column].iloc[0]]))
                    if (boolarr.sum() ==1):
                            main_genre_value = main_genres[boolarr]['main name'].values
                            if (df_clean_genre['genre 1'].iloc[index] == None):
                                    df_clean_genre['genre 1'].iloc[index] = main_genre_value
                            elif (df_clean_genre['genre 2'].iloc[index] == None and df_clean_genre['genre 1'].iloc[index]!= main_genre_value):
                                    df_clean_genre['genre 2'].iloc[index] = main_genre_value
                    #case where a sub genre belongs to more than one main genre: ex: 'Crime Comedy' (iloc[796])
                    if (boolarr.sum() ==2):
                            if (df_clean_genre['genre 1'].iloc[index] == None and df_clean_genre['genre 2'].iloc[796] == None):
                                    df_clean_genre['genre 1'].iloc[index] = main_genres[boolarr]['main name'].iloc[0]
                                    df_clean_genre['genre 2'].iloc[index] = main_genres[boolarr]['main name'].iloc[1]
    # Drop the 'Movie genres' column
    df_clean_genre = df_clean_genre.drop(columns=['Movie genres'])

    return df_clean_genre

