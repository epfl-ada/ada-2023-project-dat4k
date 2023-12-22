# ada-2023-project-dat4k
Welcome to the project dat4k


#  And if we were in a time loop ?


## Abstract : 
What if we were stuck in a time loop? This is the question we try to answer throughout this study. We conduct a temporal analysis of movies’ characteristics throughout the years and the months, focusing on the identification of patterns repeating over and over again in cinematographic history.  We aim to recognize a potential influence of the release date of a movie on its type, its plot and its success. To achieve this goal, we focus our interest on the main genres of the films, their recurring characters, the grossing profit they bring and their plot summaries. We hope that we’ll manage to give you insights on the hidden mechanisms of the cinema industry!

## Research questions :
Is there a statistically significant recurrence of specific film genres during particular seasons or months of the year?
Are there discernible patterns in the box office performance of specific film genres throughout the year, and do these patterns correlate with particular months?
Is there a relation between the connotation of the words and the season of release? For example, are there more positively connoted words in the plot of a movie in summer than in winter?
Can we predict the season or month of release of a movie if we know all of its characteristics?

A more detailed description as well as our results can be found in our [**data story**](https://jdrouet27.github.io/ada-template-website_Dat4k/).

## Additional Dataset : 

[**Budget**](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv): We will use this dataset to verify if for two similar movies the difference of box office is due to the release date or to the investment accorded to it.     

## Method : 

  - **T-test :** We compare the means of the movies with a specific genre and released in a specific month, to the movies with the same specific genre but not released in this specific month. We then check if the difference between these means is statistically different.

  - **Paired Matching :**  We use paired matching to check for causality in observed correlations. To match the two groups we standardize the continuous variables, calculate propensity scores and match, based on genre and propensity score with a threshold of >0.95. 

  - **Machine learning :** We implement KNN, random forest and logarithmic regression on our dataset. To do so first scale of the features, select the most relevant ones and split the data to obtain properly preprocessed train and test sets. We then compute the following metrics : precision, recall and F1 score to choose the best-fitting model. We finally assess our results using k-fold cross-validation. 


### Step 1 : Filtering and processing data

  1) Processing the release date column : For our annual loop study, we used movies with release month excluding those without it. Movies from years with fewer than 200 releases are also removed. 

  2) Processing the genre column : A filter and re-shaping process is done to extract more than 300 genres and categorize them into 8 main genres. We then associate each movie with up to two of these. 

  3) Processing the country column : Same as 2, we classify movies into 5 continents.

  4) Merging additional dataset: for step 4 (causal analysis), the budget needs to be taken into account, it is then merged to the main frame.



### Step 2 : Getting to know the data, visualizations 

We first show the global distribution of all genres in different seasons. Then we show the distribution of some specific genres over the months, looking for specific patterns. For example we found a pic in October for Horror movie that could be related to Halloween so we will run some others analysis.

### Step 3 : Verifying observations with hypothesis testing 

For example we separate the horror movie group in two groups: one is the horror movies in October grouped by year and the other is the horror movies released in other months than october divided by number of months and group by year.
H0 = "Number of horror movies per year in October == Mean of number of horror movies per month (excluding october)  per year." 
We used t-test to get p-value and repeated this process with other genres and other months.

### Step 4 : Causal analysis, does the film industry support the time loop ?

In Step 3, we found significant result for the Horror and Family movie genres: for particular months, their number increases. The next step is to determine the reason why there are more movies during these specific months. A possible hypothesis would be that movies are more successful during their months of predilection.

In order to verify or not this hypothesis, we will check if releasing a Horror movie in October (Halloween is 31st of October) implies a greater box-office revenue. The same analysis will be done for Family films (in July, November and December).

  **Treated** : Horror movies released in October/ Family films released in July, November, or December

  **Control** : Horror movies released in other months/ Family films not released in July, November, or December

**Confounders:**

- We think the main bias would be that box office is influenced by when the big franchise movies are released. We’re interested in the success of movies regardless of how big the franchise is.
To eliminate this bias, we take the budget as a feature.

- An important bias is the continent of release. In fact, Halloween is celebrated at different degree of popularity. For example, Russian people don't even celebrate it, whereas It is a big event in the USA. Regarding Family movies, the holiday season also depends on the continent.
For that reason we might match movies with the same continent of release.

- Finally, we take into account the release year of the movie, to eliminate to influence of inflation over the years.

A propensity score is used to match these confounders. 

With the matched data, we observe if there is a tendency that treated data has higher box office than the box office of control data averaged on the other months.


###  Step 5 : Release season estimation using machine learning  

1) Processing the plot summaries : We aim to link the temporal setting of movies with their release time by counting occurrences of specific words (e.g., winter, summer) in plot summaries. To simplify the analysis, we apply lemmatization, followed by the creation of the bag-of-words matrix. This enables us to visualize word prevalence in plots throughout the year, revealing insights into potential patterns.

In this final phase, we aim to construct a generalized model based on observed correlations between temporal aspects and movie characteristics. The objective is to predict a movie's release season (Autumn, Winter, Spring, Summer) using classifying algorithms. Our approach involves: 

  - Scaling features to prevent undue importance on larger-scale attributes (e.g., box office revenue vs. movie runtime). Encoding categorical features for distance computation in the KNN algorithm and for the logistic regression.
  - Randomly dividing the data into training and testing sets. The training set is used for model development, while the test set evaluates the model's efficiency.
  - Utilizing correlation-based feature selection to avoid over fitting and reduce computational complexity. This process retains features with the most significant variance.
  - Trying three distinct models—Boosted decision trees, logistic regression, and random forest—to find the most accurate one. We compare each model's accuracy with different hyperparameters to select the best one. 
  - Assessing and validating results using k-fold cross-validation.


## Timeline :
17.11 : Milestone P2

17.11 -- 03.12 : Pause for HW2

06.12 : Paired Matching Analysis

08.12 : ML classification algorithm

10.12 : In parallel, draft for data story

12.12 : develop website

18.12 : finalize website and data story


## Contributions of the team members 👨‍👩‍👧‍👧
<table class="tg" style="undefined;table-layout: fixed; width: 342px">
<colgroup>
<col style="width: 164px">
<col style="width: 178px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-0lax">Contributions</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">@Remidesire</td>
    <td class="tg-0lax">Pre-processed the date column<br><br>Implemented the classification algorithms<br>Developed the final text for the data story<br></td>
  </tr>
  <tr>
    <td class="tg-0lax">@julpiro</td>
    <td class="tg-0lax">Implemented the paired matching<br><br>Created meaningful visualizations<br><br>Develop the web interface</td>
  </tr>
  <tr>
    <td class="tg-0lax">@CesarCams</td>
    <td class="tg-0lax">Processed the box office revenue<br><br>Implemented the classification algorithms<br><br>Developed the final text for the data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@ninalahellec</td>
    <td class="tg-0lax">Pre-processed the genre column<br><br>Developed the final text for the data story<br><br>Implemented paired matching</td>
  </tr>
  <tr>
    <td class="tg-0lax">@jdrouet27</td>
    <td class="tg-0lax">Created meaningful visualizations<br><br>Developed t-test<br><br>Developed the web interface</td>
  </tr>
</tbody>
</table>
