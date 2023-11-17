import scipy.stats 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 
import ast
from datetime import datetime
import calendar

def ttest(df,genre, months, year_split):
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

