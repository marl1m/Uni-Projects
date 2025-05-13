import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import plotly.express as px


def autoplot(data, columns = None, purpose = str):
    """
    Automatic plot generation based on purpose given column's data type.

    Parameters:
        - purpose: str object that indicates which plot inside this function will be ran.
        - data: DataFrame, the input data containing the columns to be plotted.
        - columns: list, the list of column names to be plotted.
    """
    if purpose == 'Lifetime Expenditure':
        # Create a figure and 3x3 subplots
        fig, axs = plt.subplots(2, 4, figsize=(15, 10))

        # Flatten the subplots array for easier iteration
        axs = axs.flatten()

        # Create histograms for each subplot
        for i, (col, ax) in enumerate(zip(columns, axs)):
            # Get the data for the current column
            column_data = data[col]
    
            # Create the histogram
            ax.hist(column_data, bins= None, alpha=0.5, edgecolor='black')
    
            # Set subplot title as column name
            ax.set_title(col[15:])
    
            # Remove the x and y axis labels
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Add a main title to the plot
        fig.suptitle('Lifetime expenditure in:', fontsize=16)

        # Adjust the spacing between subplots
        fig.tight_layout()

        # Show the plot
        plt.show()

    elif purpose == 'Radar Chart Lifetime expenditures':
        # Assuming your dataset is stored in a variable called 'df'
        # Extract the mean values of the eight variables
        mean_values1 = data.loc[:,columns].mean(axis=0)

        # Apply logarithmic transformation to the mean values
        mean_values = np.log1p(mean_values1)

        # Define the variable names
        variables = ['Groceries', 'Electronics', 'Vegetables', 'Non-Alcohol Drinks',
             'Alcohol Drinks', 'Mean', 'Fish', 'Hygiene', 'Video Games']

        # Calculate the angle for each variable
        angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False).tolist()
        angles += angles[:1]  # Repeat the first angle to close the circle

        # Plot the radar chart
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, mean_values.tolist() + mean_values[:1].tolist(), alpha=0.25)
        ax.plot(angles, mean_values.tolist() + mean_values[:1].tolist(), linewidth=2)
        ax.fill(angles, [mean_values.max()]*len(angles), alpha=0.1)

        # Set the variable labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(variables)

        # Set the axis limits
        ax.set_ylim(0, mean_values.max() * 1.1)

        # Set the title
        ax.set_title('Radar Chart - Mean Values of Lifetime Spend')
        # Display the radar chart
        plt.show()

    elif purpose == 'Complaints Distribution':

        plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
        unique_values = data['number_complaints'].value_counts().index
        value_counts = data['number_complaints'].value_counts().values

        plt.bar(unique_values, value_counts, edgecolor='black')
        plt.xlabel('Number of complaints')
        plt.ylabel('Frequency')
        plt.title('Complaints Distribution')
        plt.xticks(rotation=45)  # Set x-axis tick locations and rotate labels
        plt.show()

    elif purpose == 'Age Distribution':

        plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
        unique_values = data['age'].value_counts().index
        value_counts = data['age'].value_counts().values

        plt.bar(unique_values, value_counts, edgecolor='black')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution')
        plt.xticks(rotation=45)  # Set x-axis tick locations and rotate labels
        plt.show()

    elif purpose == 'Graduation Distribution':

        plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
        unique_values = data['graduation'].value_counts().index
        value_counts = data['graduation'].value_counts()

        plt.bar(unique_values, value_counts, edgecolor='black')
        plt.xlabel('Graduation')
        plt.ylabel('Frequency')
        plt.title('Graduation Distribution')
        plt.xticks(rotation=45)  # Set x-axis tick locations and rotate labels
        plt.show()

    elif purpose == 'Minors at Home':
        # Create a figure with three subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Subplot 1 - Number of kids at home
        kids_counts = data['kids_home'].value_counts().sort_index()
        axs[0].bar(kids_counts.index.astype(int), kids_counts.values, edgecolor='black')
        axs[0].set_xlabel('Kids')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Frequency of Kids at Home')
        axs[0].set_xticks(range(int(kids_counts.index.min()), int(kids_counts.index.max()) + 1))

        # Subplot 2 - Number of teens at home
        teens_counts = data['teens_home'].value_counts().sort_index()
        axs[1].bar(teens_counts.index.astype(int), teens_counts.values, edgecolor='black')
        axs[1].set_xlabel('Teens')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Frequency of Teens at Home')
        axs[1].set_xticks(range(int(teens_counts.index.min()), int(teens_counts.index.max()) + 1))

        # Subplot 3 - Sum of kids and teens
        sum_of_both = data['minors']
        sum_counts = sum_of_both.value_counts().sort_index()
        axs[2].bar(sum_counts.index.astype(int), sum_counts.values, edgecolor='black')
        axs[2].set_xlabel('Children')
        axs[2].set_ylabel('Frequency')
        axs[2].set_title('Frequency of Minors at Home')
        axs[2].set_xticks(range(int(sum_counts.index.min()), int(sum_counts.index.max()) + 1))

        plt.tight_layout()  # Adjust the spacing between subplots
        plt.show()

    elif purpose == 'Loyalty Card':
        # Count the number of occurrences for each gender
        gender_counts = data['loyalty_card'].value_counts()

        # Create a pie chart
        label = ['No', 'Yes']
        plt.pie(gender_counts, labels = label, autopct='%1.1f%%')

        # Add a title
        plt.title('Percentage of loyalty card possession')

        # Display the plot
        plt.show()

    elif purpose == 'Gender':

        # Count the number of occurrences for each gender
        gender_counts = data['customer_gender'].value_counts()

        # Create a pie chart
        plt.pie(gender_counts, labels = ['Male', 'Female'], autopct='%1.1f%%')

        # Add a title
        plt.title('Gender Distribution')

        # Display the plot
        plt.show()
    
    elif purpose == 'Looking for outliers':

        columns = data.drop(['teens_home','kids_home','age','name','longitude',
                          'latitude','loyalty_card','typical_hour','graduation',
                          'number_complaints', 'customer_gender'], axis=1)
        num_columns = columns.shape[1]
        num_rows = math.ceil(num_columns / 5)  # Calculate the number of rows based on the number of columns

        # Calculate the height based on the number of rows
        fig_height = num_rows * 4  # Adjust the multiplier to increase or decrease the height

        # Create the figure and subplots
        fig, axs = plt.subplots(num_rows, 5, figsize=(19, fig_height))

        # Iterate through each column and create box plots
        for i, column in enumerate(columns.columns):
            row = i // 5  # Calculate the row index
            col = i % 5   # Calculate the column index
            ax = axs[row, col]  # Select the appropriate subplot
            data = columns[column]

            # Plot the box plot
            ax.boxplot(data)
            ax.set_title(column)
            ax.set_xlabel('Column')
            ax.set_ylabel('Values')

        # Hide any unused subplots
        for j in range(num_columns, num_rows * 5):
            row = j // 5  # Calculate the row index
            col = j % 5   # Calculate the column index
            fig.delaxes(axs[row, col])

        # Adjust spacing between subplots
        fig.tight_layout()

        # Display the plot
        plt.show()

    elif purpose == 'Distinct Stores Visited':
        data[['distinct_stores_visited', 'percentage_of_products_bought_promotion']].boxplot(by='distinct_stores_visited')

        # Adding labels and title
        plt.xlabel('Number of Stores Visited')
        plt.ylabel('Percentage Bought in Promotion')
        plt.title('Correlation between Stores Visited and Products Bought in Promotion')

        plt.title('Percentage of promotion bought products given the amount of stores visited')
        plt.plot(edgecolor='black')
        plt.grid(visible = None)
        plt.show()

    elif purpose == 'Time Distribution':
        # Calculate the count of each hour
        hourly_counts = data['typical_hour'].value_counts().sort_index()

        # Create the time plot
        plt.figure(figsize=(12, 6))  # Adjust the figure size as needed


        # Customize the plot appearance
        plt.plot(hourly_counts.index, hourly_counts.values, marker='o', linestyle='-')
        plt.fill_between(hourly_counts.index, hourly_counts.values,alpha=0.3)

        plt.xlabel('Typical Hour')
        plt.ylabel('Count')
        plt.title('Frequency of typical hours')

        # Adjustable dotted line density
        dotted_line_density = 5


        plt.xticks(range(24))  # Set the x-axis ticks to represent the 24-hour values
        plt.grid(True, linestyle=':', alpha=0.5)


        plt.show()

    elif purpose == 'HeatMap':
        
        heatmap_vars = data.drop(['graduation','customer_gender','latitude','longitude','loyalty_card'], axis = 1)

        corr = heatmap_vars.corr()

                # Just the upper triangle 
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Rotate x-axis labels if needed
        plt.xticks(rotation=90)

        ax = sns.heatmap(corr,
                    xticklabels=corr.columns,
                    yticklabels=corr.columns,
                    mask = mask,
                    cmap = 'GnBu')

        # Set the fontsize of the annotations
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

        # Display the plot
        plt.show()

    return plt.show()


def create_geo_map(data, lat, long):
  
    fig = px.scatter_mapbox(data, 
                        lat=lat, 
                        lon=long, 
                    
                     
                       # color_continuous_scale=color_scale,
      
                        zoom=10, 
                        height=800,
                        width=800)
 #fig.update_traces(cluster=dict(enabled=True))

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()


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

