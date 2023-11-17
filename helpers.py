import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import ast
 
def select_years(df):
    # Drop nan values
    df = df.dropna(subset=['Movie release date'])
    # Add release year column
    df['Movie release year'] = df['Movie release date'].str[0:4]
    # Convert to numeric values
    df['Movie release year'] = pd.to_numeric(df['Movie release year'], errors='raise') 
    # Sort the movies by ascending order of release year
    df = df.sort_values('Movie release year', ascending=True) 
    # Drop the first row which has an error in the release year (1010)
    df = df.drop(df[df['Movie release year'] == 1010].index)
    # Reset the indexation 
    df = df.reset_index(drop = True)

    return df

def plot_percentage_missing_month_per_year(df):
    percentage = lambda x: (x.astype(str).apply(len) < 5).mean() * 100

    missing_data_percentage = df.groupby('Movie release year')['Movie release date'].apply(percentage)

    plt.plot(missing_data_percentage.index, missing_data_percentage.values, marker='.')
    plt.title('Percentage of missing month per year')
    plt.xlabel('Year')
    plt.ylabel('Percentage missing [%]')
    plt.show()

def dataframe_with_months (df):
    # Remove the row which don't have the month of release
    df = df[df['Movie release date'].str.len() > 4]
    # Reset the indexation 
    df = df.reset_index(drop = True)
    return df

def nmbr_movie_years (df1,df2):
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

def select_main_years(df1,df2):
    film_counts_year_without_missing_months = df2['Movie release year'].value_counts().sort_index()
    years_under_200 = film_counts_year_without_missing_months.index[film_counts_year_without_missing_months.values > 200]
    df1 = df1[df1['Movie release year'].isin(years_under_200)]
    return df1

def clean_date_and_season (df):
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
  
    
def plot_average_monthly_revenue(monthly_revenue_drama, monthly_revenue_comedy, monthly_revenue_romance, 
                                 monthly_revenue_thriller, monthly_revenue_family, monthly_revenue_action, 
                                 monthly_revenue_horror, monthly_revenue_informative):
    # calculate the average revenue per month
    average_monthly_revenue_drama = monthly_revenue_drama.groupby(level=0).mean()
    average_monthly_revenue_comedy = monthly_revenue_comedy.groupby(level=0).mean()
    average_monthly_revenue_romance = monthly_revenue_romance.groupby(level=0).mean()
    average_monthly_revenue_thriller = monthly_revenue_thriller.groupby(level=0).mean()
    average_monthly_revenue_family = monthly_revenue_family.groupby(level=0).mean()
    average_monthly_revenue_action = monthly_revenue_action.groupby(level=0).mean()
    average_monthly_revenue_horror = monthly_revenue_horror.groupby(level=0).mean()
    average_monthly_revenue_informative = monthly_revenue_informative.groupby(level=0).mean()

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

    
def plot_average_monthly_revenue_percentage(monthly_revenue_drama, monthly_revenue_comedy, monthly_revenue_romance, 
                                 monthly_revenue_thriller, monthly_revenue_action, monthly_revenue_family, 
                                 monthly_revenue_horror, monthly_revenue_informative):
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

    # plot the data
    plt.plot(average_monthly_revenue_drama, label='Drama')
    plt.plot(average_monthly_revenue_comedy, label='Comedy')
    plt.plot(average_monthly_revenue_romance, label='Romance')
    plt.plot(average_monthly_revenue_thriller, label='Thriller')
    plt.plot(average_monthly_revenue_action, label='Action')
    plt.plot(average_monthly_revenue_family, label='Family')
    plt.plot(average_monthly_revenue_horror, label='Horror')
    plt.plot(average_monthly_revenue_informative, label='Informative')

    plt.xlabel('Month')
    plt.ylabel('Percentage of Box Office Revenue per Genre')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.title('Percentage of Box Office Revenue per Month for Different Genres')
    plt.legend()

    plt.xlim(1, 12)

    plt.show()

       