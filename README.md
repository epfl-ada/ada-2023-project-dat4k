# ada-2023-project-dat4k
Welcome to the project dat4k


#  And if we were in a time loop ?


## Abstract : 
What if we were stuck in a time loop? This is the question we try to answer throughout this study. We conduct a temporal analysis of movies’ characteristics throughout the years and the months, focusing on the identification of patterns repeating over and over again in cinematographic history.  We aim to recognize a potential influence of the release date of a movie on its type, its plot and its success. To achieve this goal, we focus our interest on the main genres of the films, their recurring characters, the grossing profit they bring and their plot summaries. We hope that we’ll manage to give you insights on the hidden mechanisms of the cinema industry!

## Research questions :

Is there a statistically significant recurrence of specific film genres during particular seasons or months of the year?
Are there discernible patterns in the box office performance of specific film genres throughout the year, and do these patterns correlate with particular months?
Is there a relation between the connotation of the words and the season of release? For example, are there more positively connotated words in the plot of a movie in summer than in winter?
Can we predict the season or month of release of a movie if we know all of its characteristics?

## Additional Dataset : 
[**Vocabularies**](https://drive.google.com/drive/folders/1-KcpE8cju60CcNXWc_gPZ6x3V8r7T5eH?usp=share_link):  We chose to add this dataset which is three lists of positive, negative and violent adjectives. We will use it to classify movies by a plot analysis thanks to different algorithms.


[**Budget**](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv): We will use this dataset to verify if for two similar movies the difference of box office is due to the release date or to the investment accorded to it     

## Method : 

  - **T-test :** We compare the means of the movies with a specific genre and released in a specific month, to the movies with the same specific genre but not released in this specific month. We then check if the difference between these means is statistically different.

  - **Paired Matching :**  We use paired matching to check for causality in observed correlations. To match the two groups we standardize the continuous variables, calculate propensity scores and match, based on genre and propensity score with a threshold of >0.95.


### Step 1 :

  1) Processing the release date column : For our annual loop study, we used movies with release month excluding those without it. Movies from years with fewer than 200 releases are also removed. 

  2) Processing the genre column : A filter and re-shaping process is done to extract more than 300 genres and categorize them into 8 main genres. We then associate each movie with up to two of these. 

  3) Processing the country column : Same as 2, we classify movies into 5 continents.

  4) Processing the box office revenue column : ...

  5) Processing the plot summaries : We aim to link the temporal setting of movies with their release time by counting occurrences of specific words (e.g., winter, summer) in plot summaries. To simplify analysis, we apply stemming and lemmatization, followed by the bag-of-words algorithm. This enables us to visualize word prevalence in plots throughout the year, revealing insights into potential patterns.



### Step 2 : Getting to know the data, visualizations 

We first show the global distribution of all genres in different seasons. Then we show the distribution of some specific genres over the months, looking for specific patterns. For example we found a pic in October for Horror movie that could be related to Halloween so we will run some others analysis.

### Step 3 : Verifying observations with hypothesis testing 

We separated the horror movie group in two groups: one is the horror movies in October grouped by year and the other is the horror movies released in other months than october divided by number of months and group by year.
H0 = "Number of horror movies per year in October == Mean of number of horror movies per month (excluding october)  per year." 
We used t-test to get p-value and repeated this process with other genres and other months.

### Step 4 : Box office of Horror Movie ; Paired Matching 

  - Check if having the higher box office for horror films is caused by the fact that they're released in October (check the influence of Halloween).
    **Treated** : Horror movies released in October
    **Control** : Horror movies released in other months

  We think the main biais would be that box office is influenced by when the big franchise movies are released. We’re interested in the success of horror movies in October regardless of how big the franchise is.
  To eliminate this biais, we take the budget as a feature.

  - Secondly, an important biais is the continent of release. In fact, Halloween is celebrated at different degree of popularity. For example, Russian people don't even celebrate it, whereas It is a big event in the USA.
  For that reason we might want to match movies with the same continent of release.

**Features to calculate the propensity score :**
  - Budget 
  - Country of release

With the matched data, we will then observe if there is a tendency that treated data has higher box office than the box office of control data averaged on the other months.


### Step 5 : Release season estimation using machine learning  

In this final phase, we aim to construct a generalized model based on observed correlations between temporal aspects and movie characteristics. The objective is to predict a movie's release season (Autumn, Winter, Spring, Summer) using classifying algorithms. Our approach involves: 

  - Scaling features to prevent undue importance on larger-scale attributes (e.g., box office revenue vs. movie runtime). Encoding categorical features for distance computation in the KNN algorithm.
  - Randomly dividing the data into training and testing sets. The training set is used for model development, while the test set evaluates the model's efficiency.
  - Utilizing correlation-based feature selection to avoid overfitting and reduce computational complexity. This process retains features with the most significant variance.
  - Trying three distinct models—KNN, logistic regression, and random forest—to find the most accurate one. Tuning hyperparameters for each model before the comparison of metrics like accuracy, F1 score, and ROC curve.
  - Assessing and validating results using k-fold cross-validation.


## Timeline :
17.11 : Milestone P2

17.11 -- 03.12 : Pause for HW2

06.12 : Paired Matching Analysis

08.12 : ML classification algorithm

10.12 : In parrallel, draft for data story

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
    <th class="tg-0lax">Tasks until P3</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">@Remidesire</td>
    <td class="tg-0lax">Implement the classification algorithms<br><br>Develop the web interface</td>
  </tr>
  <tr>
    <td class="tg-0lax">@julpiro</td>
    <td class="tg-0lax">Implement the paired matching<br><br>Create meaningful visualizations<br><br>Develop the web interface</td>
  </tr>
  <tr>
    <td class="tg-0lax">@CesarCams</td>
    <td class="tg-0lax">Implement the classification algorithms<br><br>Develop the final text for the data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@ninalahellec</td>
    <td class="tg-0lax">Continue working on t-test<br><br>Develop the final text for the data story<br><br>Implement paired matching</td>
  </tr>
  <tr>
    <td class="tg-0lax">@jdrouet27</td>
    <td class="tg-0lax">Create meaningful visualizations<br><br>Continue working on t-test<br><br>Develop the web interface</td>
  </tr>
</tbody>
</table>


