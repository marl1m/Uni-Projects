# Customer Segmentation: A Key to Unlocking Business Growth and Success
##### Machine Learning II - Data Science Degree - NOVA IMS

##
### Project Description
In this data science project, we aim to leverage statistical and machine learning techniques to perform customer segmentation for a retail business. The project involves analyzing two datasets - customer_info and customer_basket - containing information about customer demographics, spending habits, and purchase behavior. The goal is to identify distinct customer segments, analyze their behavior, and develop targeted marketing strategies to maximize customer engagement and loyalty.

#### Objectives:
1. Identify Relevant Customer Segments:
- Utilize unsupervised learning techniques to identify meaningful segments within the customer base based on shared characteristics.
- Develop clusters that can be used to tailor targeted marketing strategies for each segment.

2. Analyze Customer Behavior:
- Analyze the behavior of identified customer segments to gain insights into their motivations, preferences, and needs.
- Explore purchasing patterns, loyalty card usage, complaint history, and other relevant data.

3. Develop Targeted Marketing Strategies:
- Leverage the customer_basket dataset to craft personalized promotions, advertising campaigns, and product offerings for each customer segment.
- Implement creative and effective promotional strategies based on association rules derived from basket data.

#### Data Description
1. customer_info Dataset:
- Contains information on customer demographics, spend behavior, and historical transactions.
- Features include customer ID, name, birth date, number of kids and teens at home, location, loyalty card details, and spending on various product categories.

2. customer_basket Dataset:
- Comprises 80000 random baskets from customers, with transaction details and lists of purchased goods.
- Transaction-related information for the last 6 months, connected to the customer_info dataset through customer ID.

3. Additional Data:
- product_mapping.xlsx: Excel file mapping product names to categories.
##

### Repository Description
This repository contains all the files created during the development of our project. In the following paragraphs will be a short description of how the repository is organized and what each file contains:
- [Notebooks folder](notebooks): folder that contains the 3 jupyter notebooks sorted according to the project's workflow, the files constitute the project's main code, the outputs and the various visualizations that are fundamental for making customer segmentation decisions.
- [Utils folder](utils): folder that contains the 3 py files for each notebook, each with the functions needed for the main code to work, so the code is more optimized and generalizable.
- [Data folder](data): folder that contains the 2 csvs with the data used in the project and an excel file with additional product data.
- [Results folder](results): folder that contains a csv file and another folder, the csv file is the end result of customer segmentation in a simplified dataset consisting only of the name of the customer and the cluster and their ids, in the folder there are several csvs, each one being the customer_info dataset filtered by a cluster.
- [Report](Report.pdf): the academic report of the project, containing the executive summary, the methodology used in the work and the conclusions drawn.
- [Metadata](metadata.txt): text file that contains the metadata of all the used data in our project.
- [README](README.md): (this) file that contains the project and the repository descriptions.
##

### Possible Improvements
While the project successfully demonstrated effective customer segmentation for a retail business, there are several areas for potential improvement and future exploration:

1. More In-Depth Cluster Profiling
- Enhancing the depth of cluster profiling could provide a richer understanding of each customer segment. This involves delving deeper into the characteristics, behaviors, and needs of identified clusters. Conducting more granular analyses within clusters, such as sub-segmentation based on specific features, can offer a nuanced perspective and enable more targeted marketing strategies.

2. Optimization and Generalization of Code
- Further optimization and generalization of the code used for customer segmentation would contribute to improved efficiency and applicability. Implementing coding practices that enhance readability, modularity, and scalability can lead to a more robust and adaptable solution. Additionally, exploring parallel processing or distributed computing techniques may optimize the code's performance on larger datasets, ensuring scalability.

3. Utilization of Additional Product Mapping Excel
- The project introduced an additional product mapping Excel file, and further exploration of its potential impact on customer segmentation could be valuable, particularly in the realms of profiling and association rules. Integrating this mapping information into the analysis may offer a detailed understanding of how specific product categories influence customer behavior and preferences within each segment. By incorporating this mapping data into association rule mining, the project could uncover intricate patterns and relationships between different product types, providing actionable insights for targeted marketing strategies. Leveraging the additional product mapping specifically for profiling and association rules could refine the segmentation results and contribute to a more nuanced and effective customer engagement approach.
##
