import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 
import ast



def visualizing_data(df, split_year, genre):
    #reducing colums to see clearly
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
        plt.title(f'Number of {genre} movie per month for all years')
    elif when == 'b':
        plt.title(f'Number of {genre} movie per month before {split_year}')
    elif when == 'a':
        plt.title(f'Number of {genre} movie per month after {split_year}')
    plt.xlabel('Release month')
    plt.ylabel(f'ratio of {genre} movies ')
    plt.grid()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()