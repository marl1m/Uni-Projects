import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import ast

def define_graduation(cust_info):
  high_degree = cust_info[cust_info['customer_name'].str.contains('|'.join(['Bsc.', 'Msc.', 'Phd.']))].copy()
  no_degree = cust_info[~cust_info['customer_name'].str.contains('|'.join(['Bsc.', 'Msc.', 'Phd.']))].copy()

  # split the column_name column into two columns for school graduation and name
  high_degree[['graduation', 'name']] = high_degree['customer_name'].str.split(' ', n=1, expand=True)
  no_degree['name'] = no_degree['customer_name']

  high_degree = high_degree.drop('customer_name', axis = 1)
  no_degree = no_degree.drop('customer_name', axis = 1)

  df = pd.concat([high_degree, no_degree]).copy()

  last_col1 = df.pop('graduation')
  df.insert(0, 'graduation', last_col1)
  last_col2 = df.pop('name')
  df.insert(0, 'name', last_col2)
  df = df

  # encoding the graduation levels
  graduation_dict = {'Bsc.': 1, 'Msc.': 2, 'Phd.': 3}
  df['graduation'] = df['graduation'].map(graduation_dict).fillna(0)

  return df 








def calculate_age(data, birth_column):
    """
    Convert 'birth_column' column into 'age' column in a data DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing customer information.
        birth_column : Column from the data DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'birth_column' values replaced by 'age' values.
    """
    date_now = pd.to_datetime('2023-01-01 00:00:00', format ='%Y-%m-%d %H:%M:%S' )

    data[birth_column] = pd.to_datetime(data[birth_column], format = '%m/%d/%Y %I:%M %p')

    data[birth_column] = ((date_now - data[birth_column])/np.timedelta64(1,'Y')).astype('int')

    data.rename(columns = {birth_column: 'age'}, inplace = True)

    return data['age']


def dispersion(train_set,n=50):


  lista = []
  for k in range(1, n):
      kmeans = KMeans(n_clusters=k, random_state=0).fit(train_set)
      lista.append(kmeans.inertia_)
  return lista

def find_unexpected_rows(data, column: str):
  if column == 'customer_name':

   # Define the regex pattern for the expected customer_name format
    expected_pattern = r"(?:(?:Bsc\.|Msc\.|Phd\.)\s)?[A-Z][a-z]+\s[A-Z][a-z]+"

  # Create a boolean mask for the rows that don't match the expected pattern
    mask = ~data[column].str.match(expected_pattern)

  # Select the rows that don't match the expected pattern
    unexpected_rows = data[mask]

  elif column == 'customer_birthdate':

    # Define the regex pattern for the expected customer_name format
    expected_pattern = r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}\s+(?:AM|PM)"

    # Create a boolean mask for the rows that don't match the expected pattern
    mask = ~data[column].str.match(expected_pattern)

    # Select the rows that don't match the expected pattern
    unexpected_rows = data[mask]

  return unexpected_rows



def plot_dendrogram(model, **kwargs):
    '''
    Create linkage matrix and then plot the dendrogram
    Arguments: 
    - model(HierarchicalClustering Model): hierarchical clustering model.
    - **kwargs
    Returns:
    None, but dendrogram plot is produced.
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)




def visualize_data_points_grid(data, scaled_data, som_model, color_variable, color_dict):
  '''
  Plots scatter data points on top of a grid that represents
  the self-organizing map.

  Each data point can be color coded with a "target" variable and
  we are plotting the distance map in the background.

  Arguments:
  - som_model(minisom.som): Trained self-organizing map.
  - color_variable(str): Name of the column to use in the plot.

  Returns:
  - None, but a plot is shown.
  '''

  # Subset target variable to color data points
  target = data[color_variable]

  fig, ax = plt.subplots()

  # Get weights for SOM winners
  w_x, w_y = zip(*[som_model.winner(d) for d in scaled_data])
  w_x = np.array(w_x)
  w_y = np.array(w_y)

  # Plot distance back on the background
  plt.pcolor(som_model.distance_map().T, cmap='bone_r', alpha=.2)
  plt.colorbar()

  # Iterate through every data points - add some random perturbation just
  # to avoid getting scatters on top of each other.
  for c in np.unique(target):
      idx_target = target==c
      plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                  w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                  s=50, c=color_dict[c], label=c)

  ax.legend(bbox_to_anchor=(1.2, 1.05))
  plt.grid()
  plt.show()


 
  te = TransactionEncoder()
  te_fit = te.fit(train).transform(train)
  transactions_items0 = pd.DataFrame(te_fit, columns=te.columns_)

  frequent_itemsets_grocery0 = apriori(
      transactions_items0, min_support=0.05, use_colnames=True
      )

  frequent_itemsets_grocery0.sort_values(by='support', ascending=False)

  # We'll use a confidence level of 20%
  rules_grocery_cl = association_rules(frequent_itemsets_grocery0, 
                                  metric="confidence", 
                                  min_threshold=0.2)

  return rules_grocery_cl.loc[:,['antecedents','consequents','lift']].sort_values('lift')