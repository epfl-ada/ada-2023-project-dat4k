import numpy as np
import networkx as nx


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