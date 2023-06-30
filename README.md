# DataDetective App :detective:

Welcome to DataDetective, your personal data analysis assistant! DataDetective is an interactive web application that allows you to upload, investigate, clean, and analyze your data with ease. Whether you are a data scientist, analyst, or just someone curious about exploring data, DataDetective is here to help you uncover valuable insights and make data-driven decisions.

## Table of Contents

1. [Description](https://github.com/soopertramp/DataDetective_app#description)
2. [Features](https://github.com/soopertramp/DataDetective_app#features)
   - [Uploading Data](https://github.com/soopertramp/DataDetective_app#features)
   - [Data Investigation](https://github.com/soopertramp/DataDetective_app#data-investigation)
   - [Data Cleaning](https://github.com/soopertramp/DataDetective_app#data-cleaning)
   - [Data Analysis](https://github.com/soopertramp/DataDetective_app#data-analysis)
   - [Downloading Results](https://github.com/soopertramp/DataDetective_app#downloading-results)
4. [Installation](https://github.com/soopertramp/AnalytiXx_app#installation)
5. [Usage](https://github.com/soopertramp/AnalytiXx_app#usage)
6. [Examples](https://github.com/soopertramp/AnalytiXx_app#examples)
7. [Use Cases](https://github.com/soopertramp/AnalytiXx_app#use-cases)
8. [Contributing](https://github.com/soopertramp/AnalytiXx_app#contributing)
9. [License](https://github.com/soopertramp/AnalytiXx_app#license)

## Description
The Data Analysis App is a Streamlit application designed to simplify the data analysis process. It provides a user-friendly interface for uploading datasets, investigating the data, cleaning it, and generate and download the plots you created. The app aims to streamline the data analysis workflow and enable users to gain insights from their datasets quickly.

## Features

#### Uploading Data

Click on the "Browse" button to select a CSV or Excel file from your local machine.
Once the file is selected, click on the "Upload" button to start the analysis.

#### Data Investigation

After uploading the file, DataDetective performs a comprehensive investigation on the dataset and provides you with detailed information, including:
-  Number of rows and columns
-  Data types of each column
-  Summary statistics (count, mean, standard deviation, min, max, etc.)
-  Sample of the data
-  This investigation helps you understand the structure and content of your dataset.

#### Data Cleaning

DataDetective offers a set of data cleaning techniques to ensure your dataset is accurate and ready for analysis. The cleaning process includes:

-  DateTime Conversion : This is used to convert different types of data into a special data type called "datetime." In simple terms, a datetime object represents a specific date and time. It allows us to work with dates and times in a more convenient and meaningful way.
-  Handling missing values: DataDetective identifies missing values and provides options to handle them, such as imputation or removal.
-  Removing duplicates: DataDetective detects and removes duplicate rows in the dataset.
-  Handling outliers: DataDetective identifies outliers and provides options for handling them, such as removing or transforming them.
-  One Hot Encoding : One-hot encoding is a process used to convert categorical data into a format that machine learning algorithms can understand and work with more effectively.
-  Transform The Data : Standardize or Normalize
-  Convert The Column to Lowercase : This operation is useful when you want to ensure consistency in text data by converting all characters to lowercase.
-  Convert Boolean Column to Binary : Converting boolean columns to binary representation can be useful when you want to work with numerical representations of boolean variables.

The cleaned dataset is displayed, highlighting the changes made during the cleaning process.

#### Data Analysis
DataDetective provides various analysis tools and visualizations to help you gain insights from your data. Some of the analysis features include:

-  Histograms: Visualize the distribution of numerical variables.
-  Bar plots: Explore categorical variables and compare their frequencies.
-  Scatter plots: Investigate relationships between two numerical variables.
-  Correlation matrix: Examine correlations between variables.
-  Pivot tables: Generate summary statistics and cross-tabulations.
-  These analysis tools empower you to explore patterns, trends, and relationships within your data.

#### Downloading Results
DataDetective allows you to download the cleaned dataset after applying the necessary data cleaning operations. Additionally, you can download any generated analysis plots and summaries to save and share with others.

## Installation
To run the Data Analysis App locally, follow these steps:

- Clone the repository:

``` git clone https://github.com/soopertramp/DataDetective_app.git```

- Navigate to the project directory:

``` cd datadetective ```

- Install the required dependencies:

``` pip install -r requirements.txt ```

- Run the app:

```streamlit run app.py ```

### [Check out the app here](https://analytics.streamlit.app/)

## Usage

- Launch the app by running the app.py script.

- Upload a CSV or Excel file by clicking on the "Upload a file" button.

- Perform basic investigation on the dataset to understand its structure and characteristics.

- Apply data cleaning functions to preprocess the dataset according to your needs.

- Visualize the data using the provided plotting functions to gain insights.

- Download the plots.

- Download the cleaned dataset by clicking on the "Download Cleaned Data" link.

- View the download count to see how many times the cleaned dataset has been downloaded.

## Examples
Here are a few examples to demonstrate how you can utilize DataDetective:

#### Exploring Sales Data:

-  Upload a sales dataset to investigate sales trends and patterns.
-  Clean the dataset by handling missing values, removing duplicates, and formatting columns.
-  Analyze the cleaned dataset to identify top-selling products, regional sales performance, and customer segmentation.

#### Analyzing Customer Satisfaction Surveys:

-  Upload a customer survey dataset to gain insights into customer satisfaction.
-  Investigate the dataset to understand survey response distributions and demographic characteristics.
-  Clean the dataset by imputing missing values, removing outliers, and standardizing responses.
-  Analyze the cleaned dataset to identify factors influencing customer satisfaction and detect correlations between survey questions.

#### Examining Financial Data:

-  Upload financial data to investigate financial performance metrics.
-  Perform data cleaning operations such as handling missing values, removing outliers, and formatting columns.
-  Analyze the cleaned dataset to calculate key financial ratios, visualize trends in revenue and expenses, and identify anomalies.

## Contributing
Contributions to the Data Analysis App are welcome! If you encounter any issues, have suggestions for improvements, or would like to add new features, please feel free to open an issue or submit a pull request. Contributions that improve the functionality, usability, or documentation of the app are highly appreciated.

## License
The Data Analysis App is open source and is available under the MIT License. Feel free to modify and use the code for your own projects but don't forget to give the credit :knife:
