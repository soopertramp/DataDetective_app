# Data Analysis App

This is a Streamlit-based data analysis app that allows users to upload a CSV or Excel file, perform basic data investigation, clean the data, and generate a data profiling report. The app provides various functionalities for data preprocessing, visualization, and exporting the cleaned dataset.

## Table of Contents

1. Description
2. Features
3. Installation
4. Usage
5. Examples
6. Use Cases
7. Contributing
8. License

## Description
The Data Analysis App is a Streamlit application designed to simplify the data analysis process. It provides a user-friendly interface for uploading datasets, investigating the data, cleaning it, and generating a detailed data profiling report. The app aims to streamline the data analysis workflow and enable users to gain insights from their datasets quickly.

## Features
Upload CSV or Excel files: Users can upload their datasets in either CSV or Excel format.
Basic investigation: Users can explore the dataset before cleaning the data, including viewing the entire dataset, top/bottom rows, shape, column types, missing values, and duplicate values.
Data cleaning: The app provides various data cleaning functions, such as handling missing values, removing duplicates, handling outliers, converting data types, converting to lowercase, and one-hot encoding categorical variables.
Data visualization: Users can visualize the dataset using various plots and charts to gain insights and identify patterns in the data.
Data profiling report: The app generates a comprehensive data profiling report using pandas-profiling, providing detailed statistics, distributions, correlations, and interactive visualizations of the dataset.
Export cleaned dataset: Users can download the cleaned dataset in CSV format.
Download count: The app keeps track of the number of times the cleaned dataset has been downloaded.
Installation
To run the Data Analysis App locally, follow these steps:

## Clone the repository:

bash
Copy code
git clone https://github.com/your-username/data-analysis-app.git
Navigate to the project directory:

bash
Copy code
cd data-analysis-app
Install the required dependencies:

Copy code
pip install -r requirements.txt
Run the app:

arduino
Copy code
streamlit run app.py
Access the app in your web browser at http://localhost:8501.

## Usage
Launch the app by running the app.py script.
Upload a CSV or Excel file by clicking on the "Upload a file" button.
Perform basic investigation on the dataset to understand its structure and characteristics.
Apply data cleaning functions to preprocess the dataset according to your needs.
Visualize the data using the provided plotting functions to gain insights.
Generate a data profiling report by clicking on the "Generate Report" button.
Download the cleaned dataset by clicking on the "Download Cleaned Data" link.
View the download count to see how many times the cleaned dataset has been downloaded.

## Examples
Here are some examples of how the Data Analysis App can be used:

Exploratory Data Analysis (EDA): Upload a dataset, perform basic investigation, visualize the data, and generate a data profiling report to gain insights into the dataset's structure, distributions, and relationships.

Data Cleaning: Apply various data cleaning functions to handle missing values, remove duplicates, handle outliers, and transform the dataset into a clean and consistent format.

Dataset Preprocessing: Use the app to preprocess the dataset by converting data types, converting text to lowercase, and performing one-hot encoding on categorical variables.

Download Count Tracking: Track the number of times the cleaned dataset has been downloaded to monitor its popularity and usage.

## Contributing
Contributions to the Data Analysis App are welcome! If you encounter any issues, have suggestions for improvements, or would like to add new features, please feel free to open an issue or submit a pull request. Contributions that improve the functionality, usability, or documentation of the app are highly appreciated.

## License
The Data Analysis App is open source and is available under the MIT License. Feel free to modify and use the code for your own projects but don't forget to give the credit :knife:
