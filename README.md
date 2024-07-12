# Customer Segmentation using Kmeans

# Customer Segmentation Machine Learning Project

## Introduction

### Project Overview:
This project focuses on utilizing machine learning techniques, specifically K-Means clustering, for customer segmentation in a retail setting. The primary goal is to identify distinct groups of customers based on their purchasing behavior. By segmenting customers, businesses can tailor marketing strategies, improve customer retention, and optimize product offerings to better meet the diverse needs of different customer segments.

### Purpose:
The purpose of this project is twofold:
1. **Segmentation**: To divide customers into meaningful groups based on their purchasing patterns and behaviors.
2. **Insights**: To derive actionable insights from the segmentation that can inform marketing campaigns, product recommendations, and overall business strategy.

### Goals:
- Apply K-Means clustering algorithm to segment customers using transactional data.
- Evaluate the effectiveness of the segmentation through metrics like silhouette score.
- Interpret and visualize the results, customer engagement, and retention rates for each segment.

## Dataset Description

### Source:
The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset). It contains transactional data gathered over a period of [2010-2011]. The dataset was chosen for its richness in customer transaction details, which are crucial for effective customer segmentation.

### Key Attributes:
- **Invoice No**: A unique identifier for each transaction.
- **Stock Code**: Identifier for the product purchased.
- **Description**: Brief description of the product.
- **Quantity**: Number of units purchased per transaction.
- **Invoice Date**: Date and time of the transaction.
- **Unit Price**: Price per unit of the product.
- **Customer ID**: Unique identifier for each customer.
- **Country**: Country where the customer resides or where the transaction took place.

### Relevance to Customer Segmentation:
- **Purchase History**: The dataset’s transactional nature provides insights into each customer’s purchase behavior, including frequency, recency, and monetary value (RFM).
- **Customer ID**: Enables tracking of individual customer transactions, facilitating segmentation based on buying patterns.
- **Total Sales**: A critical metric for understanding customer spending habits and their value to the business.

## Methodology

### Data Cleaning
- **Handling Missing Values**:
  - Identified missing values using `df.isnull().sum()`.
  - Dropped rows with missing values in the 'Description' and 'Customer ID' columns using `dropna()`.
- **Handling Duplicates**:
  - Identified and removed duplicate rows using `df.duplicated().sum()` and `df.drop_duplicates()`.
- **Handling Negative Values**:
  - Identified and removed rows with negative values in the 'Quantity' and 'Unit Price' columns.
- **Handling Alphanumeric Stock Codes**:
  - Converted 'Stock Code' to string and removed rows with alphanumeric values using a regex pattern.
- **Handling Outliers**:
  - Detected outliers in 'Quantity' and 'Unit Price' using the Z-score method and removed them.
  - Identified date outliers in 'Invoice Date' and handled them appropriately.

### Exploratory Data Analysis (EDA)
- **Descriptive Statistics**:
  - Provided summary statistics using `df.describe()`.
- **Visualizing Sales Trends Over Time**:
  - Plotted total sales trends over time.
- **Customer Segmentation Based on RFM**:
  - Calculated Recency, Frequency, and Monetary (RFM) values for each customer.
- **Outliers Analysis in RFM**:
  - Analyzed outliers in 'Amount' and 'Frequency' using boxplots.
- **Product Analysis**:
  - Visualized top-selling products by revenue.
  - Plotted histograms for quantity distribution.

### Feature Engineering
- **Creating New Features**:
  - Extracted day of the week and month from 'Invoice Date'.
  - Calculated 'Total Sales' as the product of 'Quantity' and 'Unit Price'.
- **Encoding Categorical Variables**:
  - Applied label encoding to the 'Country' column.
- **Scaling Features**:
  - Scaled RFM features using Standard Scaler.

### Model Building
- **K-Means Clustering**:
  - Initialized and fitted K-Means with an optimal number of clusters.
  - Used the Elbow method and silhouette scores to determine the optimal number of clusters.
- **Applying PCA**:
  - Applied Principal Component Analysis (PCA) to reduce dimensionality and visualize clusters.

### Model Evaluation
- **Silhouette Score Calculation**:
  - Calculated and tracked silhouette scores for different cluster numbers to evaluate clustering performance.
- **Visualizing Clusters**:
  - Visualized clusters using PCA components.

### Evaluating Segmentation Effectiveness
- **Tracking Metrics**:
  - Tracked metrics such as average Amount, Frequency, and Recency for each cluster.
  - Identified clusters with low average amount spent for targeted strategies.
- **Visualizing Cluster Metrics**:
  - Plotted bar graphs to visualize average Amount, Frequency, and Recency by cluster.

## Summary and Conclusion

### Summary of Findings
The application of K-Means clustering to customer segmentation yielded several key insights:
- **Optimal Clustering**: Using the Elbow method and silhouette score, we determined that 5 clusters provided the best segmentation of customers.
- **Cluster Characteristics**:
  - **Cluster 0**: Customers with low frequency of purchases and low spending.
  - **Cluster 1**: Customers with high frequency of purchases and moderate spending.
  - **Cluster 2**: High-value customers with high spending and frequent purchases.
  - **Cluster 3**: New customers with recent purchases but low frequency and spending.
  - **Cluster 4**: Customers with infrequent purchases but high spending per transaction.
- **Silhouette Score**: The silhouette score of 0.61 indicates that the clusters are well-separated and distinct for 5 clusters. After interchanging the number of clusters to 2, the silhouette score improved to 0.92, indicating the clusters are even more distinct and well-separated than 5 clusters. However, I have chosen the number of clusters as 5 because it’s well optimal and balanced.
- **RFM Analysis**: Recency, Frequency, and Monetary (RFM) metrics were crucial in identifying customer segments, with high-value customers contributing significantly to the business's revenue.

### Business Insights
The segmentation analysis provided several practical insights that can drive business decisions:
- **Targeted Marketing**:
  - Cluster 2 (high-value customers) should be the focus of loyalty programs and exclusive offers to retain their business.
  - Cluster 1 and Cluster 4 customers can be targeted with campaigns to increase purchase frequency.
- **Resource Allocation**:
  - Allocate more resources to retain high-value customers while developing strategies to convert new and infrequent customers (Cluster 0 and Cluster 3) into regular buyers.
- **Product Recommendations**:
  - Tailor product recommendations based on the spending patterns of different clusters to enhance customer satisfaction and increase sales.
- **Promotional Strategies**:
  - Design specific promotions for each cluster to maximize engagement and conversion rates. For instance, offer discounts to Cluster 3 to incentivize repeat purchases.

### Limitations and Future Work
Despite the successful segmentation, the project encountered some limitations:
- **Data Quality**:
  - Missing values and outliers in the dataset required significant cleaning efforts. Future work could involve improving data collection processes to minimize these issues.
- **Feature Selection**:
  - The analysis primarily relied on RFM metrics. Incorporating additional features such as customer demographics and behavioral data could enhance segmentation accuracy.
- **Clustering Algorithm**:
  - While K-Means provided meaningful segments, exploring other clustering algorithms like DBSCAN or hierarchical clustering could yield different insights and potentially more accurate segmentation.
- **Temporal Dynamics**:
  - The current analysis does not account for changes in customer behavior over time. Future work could involve a time-series analysis to understand how customer segments evolve.

In conclusion, this project successfully applied K-Means clustering to segment customers, providing valuable insights for targeted marketing and strategic decision-making. By addressing the identified limitations and exploring additional methodologies, future work can further enhance the effectiveness of customer segmentation.

## Appendix
- **Technical Details**: For detailed technical implementation, refer to the [Google Colab Notebook](https://colab.research.google.com/drive/1J5thvCZNFlYw0scym5tB24FDA3Rx8G1Q?usp=sharing). It includes comprehensive code snippets for data cleaning, feature engineering, model building (K-Means clustering and PCA), and evaluation (silhouette score calculation and cluster visualization).
- **References**: Customer Segmentation Dataset. Retrieved from [Kaggle](https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset).

---


