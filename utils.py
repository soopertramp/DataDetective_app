# Importing Necessary Libraries
import base64
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer

import streamlit as st

# Initialize download count
def initialize_download_count() -> int:
    """
    Initialize the download count.

    This function reads the download count from a file if it exists,
    otherwise, it sets the download count to 0.

    Returns:
        The initialized download count (integer).

    Raises:
        FileNotFoundError: If the file containing the download count does not exist.

    """

    # Read the download count from the file if it exists
    try:
        with open("download_count.txt", "r") as file:
            download_count = int(file.read())
    except FileNotFoundError:
        raise FileNotFoundError("The file containing the download count does not exist.")

    return download_count

# Increment download count
def increment_download_count() -> None:
    """
    Increment the download count by 1 and store the updated count in a file.

    This function increments the download count by 1, stores the updated count in a file called "download_count.txt".

    Returns:
        None.

    """
    download_count = initialize_download_count()
    download_count += 1
    # Store the updated download count in the file
    with open("download_count.txt", "w") as file:
        file.write(str(download_count)) 
        
    return download_count     
            
# Upload CSV or Excel file and generate report
def upload_file() -> pd.DataFrame:
    """
    Upload a CSV or Excel file and generate a data profiling report.

    This function allows the user to upload a CSV or Excel file and generates a data profiling report using pandas-profiling.
    It provides options to display the report, export it as HTML, increment the download count, and provide a download link.

    Returns:
        pd.DataFrame: The uploaded DataFrame.
    """

    st.subheader('Upload a CSV or Excel file to analyze.')

    # Create a file uploader widget
    uploaded_file = st.file_uploader("### Upload a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
    # Read the uploaded file as a pandas DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            # Get the available sheets in the Excel file
            excel_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            
            # Display the sheet names and let the user choose one
            sheet_names = list(excel_sheets.keys())
            
            if len(sheet_names) == 1:
                selected_sheet = sheet_names[0]
            else:    
                selected_sheet = st.selectbox("Select sheet", sheet_names)
            
            # Read the selected sheet as a DataFrame
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        else:
            st.error("Invalid file format. Please upload a CSV or Excel file.")
            return None
        
        st.success("Your data is loaded successfully")
        return df  
                      
def basic_investigation(df: pd.DataFrame):
    """
    Perform basic investigation of the dataset before cleaning the data.

    Parameters:
        df (pd.DataFrame): The input DataFrame to investigate.
    """

    st.subheader('Basic Investigation Of The Dataset Before Cleaning The Data')

    # Display the entire dataset
    if st.button('Show Dataset', key='show_data_before_cleaning'):
        st.write('###### The display of your dataset in a tabular format, allows you to visually explore and analyze its contents, such as numerical values, categorical variables, and textual data.')
        st.subheader('Your Dataset')
        st.write(df)

    # Display the top 5 rows of the dataset
    if st.button('Show Top 5 Rows', key='show_top5_before_cleaning'):
        st.write('###### The display of the first 5 rows of your dataset')
        st.subheader('Top 5 Rows')
        st.write(df.head(5))

    # Display the bottom 5 rows of the dataset
    if st.button('Show Bottom 5 Rows', key='show_bottom5_before_cleaning'):
        st.write('###### The display of the bottom 5 rows of your dataset')
        st.subheader('Bottom 5 Rows')
        st.write(df.tail(5))

    # Display the shape of the dataset
    if st.button('Show Shape of the Dataset', key='show_shape_before_cleaning'):
        st.subheader('Shape of the dataset')
        st.write(f'The shape of your dataset is {df.shape[0]} rows and {df.shape[1]} columns')

    # Display the types of columns
    if st.button('Show Types of Columns', key='show_datatypes_before_cleaning'):
        st.subheader('The data types of your columns')
        st.write(df.dtypes)

    # Display the outliers
    if st.button('Find Outliers', key='find_outliers'):
        st.subheader('Outliers Detection Using IQR Method')

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        outliers_summary = []

        for column in numeric_columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Find outliers outside the bounds
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

            outliers_summary.append({
                'Column': column,
                'Number of Outliers': len(outliers),
                'Outliers': outliers
            })

        for outlier_info in outliers_summary:
            st.write(f"##### Outliers in column '{outlier_info['Column']}':")
            st.write(f"Number of outliers detected: {outlier_info['Number of Outliers']}")
            if outlier_info['Number of Outliers'] > 0:
                st.table(outlier_info['Outliers'])
              
    # Display descriptive statistics
    if st.button('Show Descriptive Statistics', key='show_stats_before_cleaning'):
        st.write("###### This function calculates various statistical measures, including count, mean, standard deviation, minimum value, quartiles, and maximum value for each numerical column in the DataFrame.")
        st.write("###### It helps you get a quick overview of the central tendency, dispersion, and distribution of the data in those columns.")
        st.subheader('Descriptive Statistics of Your Dataset')
        st.write(df.describe())
        
def apply_data_cleaning(df):
    """
    Apply data cleaning operations to the DataFrame based on user inputs.

    Args:
        df (pd.DataFrame): The DataFrame to apply data cleaning operations on.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    if df is None:
        st.error("Please Upload The Data, To See The Data Cleaning Inputs")
        return None
    
    st.subheader('Data Cleaning Inputs')
    
    def convert_to_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Converts specified columns in a DataFrame to datetime format.

        Parameters:
            - df (pd.DataFrame): The DataFrame to be converted.
            - columns (List[str]): A list of column names to convert to datetime.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns converted to datetime format.
        """
        st.write('#### DateTime Conversion')
        st.write('###### This is used to convert different types of data into a special data type called "datetime." In simple terms, a datetime object represents a specific date and time. It allows us to work with dates and times in a more convenient and meaningful way.')
        date_columns = st.multiselect('Columns to convert to datetime', columns, key='convert_to_datetime_multiselect')
        for column in date_columns:
            df[column] = pd.to_datetime(df[column])
        return df 
    
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values in a DataFrame based on user-selected actions.

        Parameters:
            - df (pd.DataFrame): The DataFrame to handle missing values.

        Returns:
            pd.DataFrame: The DataFrame with missing values handled according to the user-selected action.
        """
        st.write('#### How Do You Want To Handle The Missing Values?')
        st.write('###### 1. "Do Nothing": No action will be taken on the missing values.')
        st.write('###### 2. "Dropna": Rows containing missing values will be dropped from the dataset.')
        st.write('###### 3. "Fillna": You can enter a value to fill in the missing values. The input should be a numeric value, and it will be used to replace the missing values in the dataset.')
        st.write('###### 4. "Replace": You can provide replacement values for each column individually. This allows you to specify different values to replace the missing values in each column.')
        st.write('###### 5. "Simple Imputer": You can choose an imputation strategy, such as mean, median, or most frequent, to fill in the missing values. Numeric and categorical columns can be imputed separately.')
        st.write('###### 6. "KNN Imputer": This method uses the K-Nearest Neighbors algorithm to impute missing values. You can specify the number of neighbors to consider. Numeric and categorical columns can be imputed separately.')
        missing_values_action = st.selectbox('Choose TheMissing Values Action', ['Do Nothing', 'Dropna', 'Fillna', 'Replace', 'Simple Imputer', 'KNN Imputer'], key='missing_values_action')
        
        if missing_values_action == 'Do Nothing':
            # Do nothing, no action taken on missing values
            pass
        elif missing_values_action == 'Dropna':
            df = df.dropna()
        elif missing_values_action == 'Fillna':
            fill_value = st.text_input('Enter the fill value', '')
            if fill_value and fill_value.strip():
                try:
                    fill_value = float(fill_value)
                except ValueError:
                    st.error("Invalid fill value. Please provide a numeric value.")
                    st.stop()
                
                num_columns = df.select_dtypes(include=['number', 'float']).columns
                for column in num_columns:
                    df[column] = df[column].fillna(fill_value)
            else:
                st.error("Invalid fill value. Please provide a non-empty value.")
                st.stop()
        elif missing_values_action == 'Replace':
            column_names = df.columns
            replacements = {}
            for column in column_names:
                replacement_value = st.text_input(f'Enter the replacement value for {column}', '')
                replacements[column] = replacement_value
            df = df.replace(replacements)
        elif missing_values_action == 'Simple Imputer':
            imputation_strategy = st.selectbox('Imputation Strategy', ['mean', 'median', 'most_frequent'])
            numeric_columns = df.select_dtypes(include='number').columns
            categorical_columns = df.select_dtypes(exclude='number').columns
            
            if len(numeric_columns) > 0:
                selected_numeric_columns = st.multiselect('Select numeric columns for imputation', list(numeric_columns))
                if len(selected_numeric_columns) > 0:
                    numeric_imputer = SimpleImputer(strategy=imputation_strategy)
                    df[selected_numeric_columns] = numeric_imputer.fit_transform(df[selected_numeric_columns])
            
            if len(categorical_columns) > 0:
                selected_categorical_columns = st.multiselect('Select categorical columns for imputation', list(categorical_columns))
                if len(selected_categorical_columns) > 0:
                    categorical_imputer = SimpleImputer(strategy='most_frequent')
                    df[selected_categorical_columns] = categorical_imputer.fit_transform(df[selected_categorical_columns])
        
        elif missing_values_action == 'KNN Imputer':
            n_neighbors = st.number_input('Number of Neighbors', min_value=1, step=1)
            numeric_columns = df.select_dtypes(include='number').columns
            categorical_columns = df.select_dtypes(exclude='number').columns
            
            if len(numeric_columns) > 0:
                selected_numeric_columns = st.multiselect('Select numeric columns for imputation', list(numeric_columns))
                if len(selected_numeric_columns) > 0:
                    numeric_imputer = KNNImputer(n_neighbors=n_neighbors)
                    df[selected_numeric_columns] = numeric_imputer.fit_transform(df[selected_numeric_columns])
            
            if len(categorical_columns) > 0:
                selected_categorical_columns = st.multiselect('Select categorical columns for imputation', list(categorical_columns))
                if len(selected_categorical_columns) > 0:
                    categorical_imputer = SimpleImputer(strategy='most_frequent')
                    df[selected_categorical_columns] = categorical_imputer.fit_transform(df[selected_categorical_columns])
        
        else:
            st.error("Invalid missing values action. Please choose a valid action.")
            st.stop()
            
        return df
    
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicates from a DataFrame based on user selection.

        Parameters:
            - df (pd.DataFrame): The DataFrame to remove duplicates from.

        Returns:
        pd.DataFrame: The DataFrame with duplicates removed if the user selects the option, otherwise returns the original DataFrame.
        """
        st.write('#### Remove The Duplicates')
        st.write('###### This function is used to remove duplicate rows from the dataset.')
        st.write('###### Imagine you have a table of data where some rows are identical or have the same values in all columns. This can happen when you have duplicate entries or when merging or appending multiple datasets.')
        remove_duplicates = st.checkbox('Remove duplicates')
        if remove_duplicates:
            df = df.drop_duplicates()
        return df
    
    def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles outliers in a DataFrame based on user-selected actions.

        Parameters:
            - df (pd.DataFrame): The DataFrame to handle outliers.

        Returns:
        pd.DataFrame: The DataFrame with outliers handled according to the user-selected action.
        """
        st.write('#### How Do You Want To Handle The Outliers?')
        st.write('###### 1. "Box Plot": Using the interquartile range (IQR) method, outliers are identified and removed from the dataset. Values outside the lower and upper bounds, calculated using the IQR and a multiplier, are considered outliers and filtered out.')
        st.write('###### 2. "Standard Deviation": You can specify the number of standard deviations from the mean as a threshold. Values outside this range are considered outliers and filtered out from the dataset.')
        st.write('###### 3. "Z Score": You can specify a threshold value for the Z score. Values with Z scores exceeding this threshold are considered outliers and removed from the dataset. The Z score measures how many standard deviations a value is away from the mean.')
        remove_outliers = st.selectbox('Remove outliers', ['Do Nothing', 'Box Plot', 'Standard Deviation', 'Z Score'])
        num_columns = df.select_dtypes(include=['number']).columns
        
        if len(num_columns) > 0:
            if remove_outliers == 'Box Plot':
                for column in num_columns:
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
            elif remove_outliers == 'Standard Deviation':
                num_std_devs = st.number_input('Number of Standard Deviations', value=0.0, min_value=0.0, max_value=None, step=0.1)
                for column in num_columns:
                    mean = df[column].mean()
                    std = df[column].std()
                    lower_bound = mean - num_std_devs * std
                    upper_bound = mean + num_std_devs * std
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
            elif remove_outliers == 'Z Score':
                threshold = st.number_input('Z Score Threshold', value=0.0, min_value=0.0, step=0.1)
                for column in num_columns:
                    z_scores = (df[column] - df[column].mean()) / df[column].std()
                    df = df[abs(z_scores) <= threshold]
        
        return df
    
    def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs one-hot encoding on specified categorical columns of a DataFrame.

        Parameters:
            - df (pd.DataFrame): The DataFrame to perform one-hot encoding.

        Returns:
        pd.DataFrame: The DataFrame with specified categorical columns one-hot encoded.
        """
        st.write('#### One Hot Encoding')
        st.write('###### One-hot encoding is a process used to convert categorical data into a format that machine learning algorithms can understand and work with more effectively.')
        st.write('###### Imagine you have a dataset that includes a categorical feature, such as "color," with possible values like "red," "blue," and "green." Machine learning algorithms typically work with numerical data, so we need to convert these categorical values into numbers.')
        st.write('###### One-hot encoding does this conversion by creating new binary columns for each unique category in the original categorical feature. Each binary column represents a specific category and contains a 1 if the original data point belongs to that category, or 0 if it does not.')
        categorical_columns = st.multiselect('Categorical columns to one-hot encode', df.select_dtypes(include=['object']).columns)
        df = pd.get_dummies(df, columns=categorical_columns, drop_first = True)
        return df
    
    def transform_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the numerical data in a DataFrame based on user-selected method.

        Parameters:
            - df (pd.DataFrame): The DataFrame to transform.

        Returns:
        pd.DataFrame: The DataFrame with numerical columns transformed according to the user-selected method.
        """
        st.write('#### Transform The Data (Standardize or Normalize)')
        st.write('###### "Standardize": You can select numerical columns from the dataset to be standardized. Standardization involves subtracting the mean and dividing by the standard deviation for each selected column, resulting in a standardized distribution with a mean of 0 and a standard deviation of 1.')
        st.write('###### "Normalize": You can select numerical columns from the dataset to be normalized. Normalization scales the values of each selected column to a range between 0 and 1. This is achieved by subtracting the minimum value and dividing by the range (maximum value minus minimum value) for each column.')
        transformation_method = st.selectbox('Select Transformation Method', ['Standardize', 'Normalize'])
        numerical_columns = df.select_dtypes(include=['number']).columns
        
        if transformation_method == 'Standardize':
            standardize_columns = st.multiselect('Numerical columns to standardize', numerical_columns)
            for column in standardize_columns:
                df[column] = (df[column] - df[column].mean()) / df[column].std()
        
        elif transformation_method == 'Normalize':
            normalize_columns = st.multiselect('Numerical columns to normalize', numerical_columns)
            for column in normalize_columns:
                df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        
        return df
    
    def convert_to_lowercase(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts specified columns of a DataFrame to lowercase.

        Parameters:
            - df (pd.DataFrame): The DataFrame to convert columns to lowercase.

        Returns:
        pd.DataFrame: The DataFrame with specified columns converted to lowercase.
        """
        st.write('#### Convert The Column to Lowercase')
        st.write('###### This operation is useful when you want to ensure consistency in text data by converting all characters to lowercase. It can help with tasks such as text matching, grouping, or standardization.')
        lowercase_columns = st.multiselect('Columns to convert to lowercase', df.select_dtypes(include=['object']).columns)
        for column in lowercase_columns:
            df[column] = df[column].str.lower()
        return df
    
    def convert_boolean_to_binary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts specified boolean columns of a DataFrame to binary representation.

        Parameters:
            - df (pd.DataFrame): The DataFrame to convert boolean columns to binary.

        Returns:
        pd.DataFrame: The DataFrame with specified boolean columns converted to binary.
        """
        st.write('#### Convert Boolean Column to Binary')
        st.write('###### Converting boolean columns to binary representation can be useful when you want to work with numerical representations of boolean variables. In binary format, False is represented as 0 and True is represented as 1. This conversion enables you to perform numerical operations or apply machine learning algorithms that require numerical inputs.')
        boolean_columns = st.multiselect('Boolean columns to convert to binary', df.select_dtypes(include=['bool']).columns)
        for column in boolean_columns:
            df[column] = df[column].astype(int)
        return df
    
    # Perform data analysis actions
    df = convert_to_datetime(df, df.columns)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = one_hot_encode(df)
    df = transform_data(df)
    df = convert_to_lowercase(df)
    df = convert_boolean_to_binary(df)
    
    return df
        
def display_after_cleaning(df):
    """
    Display basic investigation of the dataset after cleaning.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        None.
    """
    
    st.subheader('Basic Investigation Of The Dataset After Cleaning')

    # Display the entire dataset
    if st.button('Show Dataset', key='show_dataset'):
        st.write('###### The display of your dataset in a tabular format, allows you to visually explore and analyze its contents, such as numerical values, categorical variables, and textual data.')
        st.subheader('Your Dataset')
        st.write(df)

    # Display the top 5 rows of the dataset
    if st.button('Show Top 5 Rows', key='show_top_rows'):
        st.write('###### The display of the first 5 rows of your dataset')
        st.subheader('Top 5 Rows')
        st.write(df.head(5))

    # Display the bottom 5 rows of the dataset
    if st.button('Show Bottom 5 Rows', key='show_bottom_rows'):
        st.write('###### The display of the bottom 5 rows of your dataset')
        st.subheader('Bottom 5 Rows')
        st.write(df.tail(5))

    # Display the shape of the dataset
    if st.button('Show Shape of the Dataset', key='show_dataset_shape'):
        st.subheader('Shape of the dataset')
        st.write(f'The shape of your dataset is {df.shape[0]} rows and {df.shape[1]} columns')

    # Display the types of columns
    if st.button('Show Types of Columns', key='show_dataset_columns'):
        st.subheader('The Data Types of your columns')
        st.write(df.dtypes)

    # Display missing values and duplicate values
    if st.button('Show Missing Values and Duplicate Values', key='show_dataset_missing'):
        st.subheader('Missing Values and Duplicate Values In Your Dataset')
        st.write(f"##### The DataFrame contains the following missing values.")
        missing_values = df.isnull().sum()
        missing_values_table = pd.DataFrame({'Column': missing_values.index, 'Missing Values': missing_values.values})
        st.table(missing_values_table)
        st.write(f'##### The number of duplicated rows in your dataset is {df.duplicated().sum()}')

    # Display descriptive statistics
    if st.button('Show Descriptive Statistics', key='show_dataset_statistics'):
        st.write("###### This function calculates various statistical measures, including count, mean, standard deviation, minimum value, quartiles, and maximum value for each numerical column in the DataFrame.")
        st.write("###### It helps you get a quick overview of the central tendency, dispersion, and distribution of the data in those columns.")
        st.subheader('Descriptive Statistics of Your Dataset')
        st.write(df.describe())  
            
def generate_download_link(df, file_name):
    """
    Generate a download link for the cleaned dataset.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        file_name (str): The desired name for the downloaded file.

    Returns:
        None.
    """
    if df is None:
        st.error("Please Wait For The Data To Get Cleaned So That You Can Download")
        return None
    
    st.subheader('Download Your Cleaned Dataset')

    # Convert DataFrame to CSV file
    csv = df.to_csv(index=False)

    # Generate download link
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)   
                
def plot_data(df):
    """
    Perform basic data visualization based on user-selected plot type.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        None.
    """
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('Plots')
    
    plot_types = ['Scatter Plot', 'Line Plot', 'Bar Plot', 'Histogram', 'Box Plot', 'Pie Plot']
    aggregation_levels = ['Yearly', 'Monthly', 'Daily']

    # Select plot type
    selected_plot = st.selectbox('Select plot type', plot_types)

    # Select aggregation level
    selected_aggregation = st.selectbox('Select aggregation level', aggregation_levels)

    if selected_plot == 'Scatter Plot':
        st.write('###### Scatter Plot: ')
        st.write("###### Definition : A scatter plot displays the relationship between two numerical variables. Each point on the plot represents an observation in the dataset.")
        st.write("###### Example: A scatter plot can be used to visualize the relationship between a students study time and their exam scores. Each point on the plot represents a student, with the x-axis representing study time and the y-axis representing exam scores.")

        st.subheader('Scatter Plot')
        x_variable = st.selectbox('Select x-axis variable', df.columns)
        y_variable = st.selectbox('Select y-axis variable', df.columns)

        # Check if x_variable is a date column
        if df[x_variable].dtype == 'datetime64[ns]':
            if selected_aggregation == 'Yearly':
                df['Year'] = df[x_variable].dt.year
                df_grouped = df.groupby('Year')[y_variable].mean()  # Aggregate by yearly mean
                x_values = df_grouped.index
                y_values = df_grouped.values
            elif selected_aggregation == 'Monthly':
                df['Month'] = df[x_variable].dt.to_period('M')
                df_grouped = df.groupby('Month')[y_variable].mean()  # Aggregate by monthly mean
                x_values = df_grouped.index.to_timestamp()  # Convert period index back to timestamp
                y_values = df_grouped.values
            elif selected_aggregation == 'Daily':
                df['Date'] = df[x_variable].dt.date
                df_grouped = df.groupby('Date')[y_variable].mean()  # Aggregate by daily mean
                x_values = df_grouped.index
                y_values = df_grouped.values
        else:
            # Check if x_variable is a numerical column
            if np.issubdtype(df[x_variable].dtype, np.number):
                bin_values = np.histogram(df[x_variable], bins='auto')[1]
                x_values = pd.cut(df[x_variable], bins=bin_values, labels=False)
                y_values = df[y_variable]
            else:
                # Assign numerical values to categories
                categories = df[x_variable].unique()
                category_dict = {category: i for i, category in enumerate(categories)}
                x_values = df[x_variable].map(category_dict)
                y_values = df[y_variable]

        # Download the plot
        export_button = st.button(f'Generate and Download {selected_plot}')
        if export_button:
            # Save the plot as a file
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x_values, y_values)
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title('Scatter Plot')
            plt.tight_layout()
            plt.xticks(rotation=45)
            if df[x_variable].dtype == 'datetime64[ns]':
                if selected_aggregation == 'Yearly':
                    plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired
                elif selected_aggregation == 'Monthly':
                    plt.xticks(x_values, x_values.strftime('%Y-%m'))  # Format x-axis ticks as desired
                elif selected_aggregation == 'Daily':
                    plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired

            plt.savefig(f"{selected_plot}.png")
            st.pyplot(fig)
            
            # Increment the download count
            increment_download_count()
            
            # Provide a download link for the plot
            with open(f"{selected_plot}.png", 'rb') as file:
                b64 = base64.b64encode(file.read()).decode()  # Encode the file content in base64
                href = f'<a href="data:image/png;base64,{b64}" download="{selected_plot}.png">Download {selected_plot}</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.write(":arrow_up: Click Above To Download The Plot, Thank You! :pray:")

    elif selected_plot == 'Line Plot':
        st.write('###### Line Plot: ')
        st.write("###### Definition : A line plot shows the trend or progression of a variable over a continuous interval or time.")
        st.write("###### Example: A line plot can be used to track the stock prices of a company over a period of months. The x-axis represents the time (months), and the y-axis represents the stock prices.")

        st.subheader('Line Plot')
        x_variable = st.selectbox('Select x-axis variable', df.columns)
        y_variable = st.selectbox('Select y-axis variable', df.columns)

        # Check if x_variable is a date column
        if df[x_variable].dtype == 'datetime64[ns]':
            if selected_aggregation == 'Yearly':
                df_grouped = df.groupby(df[x_variable].dt.year).mean()
                x_values = df_grouped.index
                y_values = df_grouped[y_variable].values
            elif selected_aggregation == 'Monthly':
                df_grouped = df.groupby(pd.Grouper(key=x_variable, freq='M')).mean()
                x_values = df_grouped.index
                y_values = df_grouped[y_variable].values
            elif selected_aggregation == 'Daily':
                df_grouped = df.groupby(df[x_variable].dt.date).mean()  # Aggregate by daily mean
                x_values = df_grouped.index
                y_values = df_grouped[y_variable].values
        else:
            # Check if x_variable is a numerical column
            if np.issubdtype(df[x_variable].dtype, np.number):
                x_values = df[x_variable]
                y_values = df[y_variable]
            else:
                # Assign numerical values to categories
                categories = df[x_variable].unique()
                category_dict = {category: i for i, category in enumerate(categories)}
                x_values = df[x_variable].map(category_dict)
                y_values = df[y_variable]

        # Download the plot
        export_button = st.button(f'Generate and Download {selected_plot}')
        if export_button:
            # Save the plot as a file
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_values, y_values)
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title('Line Plot')
            plt.tight_layout()
            plt.xticks(rotation=45)
            if df[x_variable].dtype == 'datetime64[ns]':
                if selected_aggregation == 'Yearly':
                    plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired
                elif selected_aggregation == 'Monthly':
                    plt.xticks(x_values, x_values.strftime('%Y-%m'))  # Format x-axis ticks as desired
                elif selected_aggregation == 'Daily':
                    plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired

            plt.savefig(f"{selected_plot}.png")
            st.pyplot(fig)
            
            # Increment the download count
            increment_download_count()

            # Provide the file for download
            with open(f"{selected_plot}.png", 'rb') as file:
                b64 = base64.b64encode(file.read()).decode()  # Encode the file content in base64
                href = f'<a href="data:image/png;base64,{b64}" download="{selected_plot}.png">Click here to download the {selected_plot}</a>'
                st.markdown(href, unsafe_allow_html=True)

    elif selected_plot == 'Bar Plot':
        st.write('###### Bar Plot: ')
        st.write("###### Definition : A bar plot represents categorical data with rectangular bars. The height of each bar corresponds to the frequency or value associated with each category.")
        st.write("###### Example: A bar plot can be used to compare the sales performance of different products in a store. Each bar represents a product, and the height of the bar represents its sales quantity or revenue.")

        st.subheader('Bar Plot')
        x_variable = st.selectbox('Select x-axis variable', df.columns)
        y_variable = st.selectbox('Select y-axis variable', df.columns)

        # Check if x_variable is a date column
        if df[x_variable].dtype == 'datetime64[ns]':
            if selected_aggregation == 'Yearly':
                df['Year'] = df[x_variable].dt.year
                df_grouped = df.groupby('Year')[y_variable].sum()  # Aggregate by yearly sum
                x_values = df_grouped.index
                y_values = df_grouped.values
            elif selected_aggregation == 'Monthly':
                df['Month'] = df[x_variable].dt.to_period('M')
                df_grouped = df.groupby('Month')[y_variable].sum()  # Aggregate by monthly sum
                x_values = df_grouped.index.to_timestamp()  # Convert period index back to timestamp
                y_values = df_grouped.values
            elif selected_aggregation == 'Daily':
                df['Date'] = df[x_variable].dt.date
                df_grouped = df.groupby('Date')[y_variable].sum()  # Aggregate by daily sum
                x_values = df_grouped.index
                y_values = df_grouped.values
        else:
            # Check if x_variable is a numerical column
            if np.issubdtype(df[x_variable].dtype, np.number):
                bin_values = np.histogram(df[x_variable], bins='auto')[1]
                x_values = pd.cut(df[x_variable], bins=bin_values, labels=False)
                y_values = df[y_variable]
            else:
                # Assign numerical values to categories
                categories = df[x_variable].unique()
                category_dict = {category: i for i, category in enumerate(categories)}
                x_values = df[x_variable].map(category_dict)
                y_values = df[y_variable]

        # Download the plot
        export_button = st.button(f'Generate and Download {selected_plot}')
        if export_button:
            # Save the plot as a file
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(x_values, y_values)
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title('Bar Plot')
            plt.tight_layout()
            plt.xticks(rotation=45)
            if df[x_variable].dtype == 'datetime64[ns]':
                if selected_aggregation == 'Yearly':
                    plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired
                elif selected_aggregation == 'Monthly':
                    plt.xticks(x_values, x_values.strftime('%Y-%m'))  # Format x-axis ticks as desired
                elif selected_aggregation == 'Daily':
                    plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired

            plt.savefig(f"{selected_plot}.png")
            st.pyplot(fig)
            
            # Increment the download count
            increment_download_count()

            # Provide a download link for the plot
            with open(f"{selected_plot}.png", 'rb') as file:
                b64 = base64.b64encode(file.read()).decode()  # Encode the file content in base64
                href = f'<a href="data:image/png;base64,{b64}" download="{selected_plot}.png">Click here to download the {selected_plot}</a>'
                st.markdown(href, unsafe_allow_html=True)

    elif selected_plot == 'Histogram':
        st.write('###### Histogram: ')
        st.write("###### Definition : A histogram is used to visualize the distribution of a continuous variable by dividing it into intervals (bins) and representing the frequency or proportion of observations within each bin.")
        st.write("###### Example: A histogram can be used to examine the distribution of students' heights in a classroom. The x-axis represents height intervals, and the y-axis represents the frequency or proportion of students within each height interval.")

        st.subheader('Histogram')
        variable = st.selectbox('Select variable', df.columns)

        # Download the plot
        export_button = st.button(f'Generate and Download {selected_plot}')
        if export_button:
            # Save the plot as a file
            # Check if variable is a numerical column
            if np.issubdtype(df[variable].dtype, np.number):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df[variable], bins='auto')
                ax.set_xlabel(variable)
                ax.set_ylabel('Frequency')
                ax.set_title('Histogram')
                plt.tight_layout()
                plt.xticks(rotation=45)
                plt.savefig(f"{selected_plot}.png")
                st.pyplot(fig)
                
                # Increment the download count
                increment_download_count()
                
                # Provide a download link for the plot
                with open(f"{selected_plot}.png", 'rb') as file:
                    b64 = base64.b64encode(file.read()).decode()  # Encode the file content in base64
                    href = f'<a href="data:image/png;base64,{b64}" download="{selected_plot}.png">Click here to download the {selected_plot}</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.write('Selected variable is not numerical.')

    elif selected_plot == 'Box Plot':
        st.write('###### Box Plot: ')
        st.write("###### Definition : A box plot (or box-and-whisker plot) displays the summary statistics of a numerical variable, including the median, quartiles, and outliers.")
        st.write("###### Example: A box plot can be used to compare the distribution of salaries across different job positions in a company. Each box represents a job position, and the vertical lines (whiskers) extend to show the range of salaries.")

        st.subheader('Box Plot')
        x_variable = st.selectbox('Select x-axis variable', df.columns)
        y_variable = st.selectbox('Select y-axis variable', df.columns)

        # Download the plot
        export_button = st.button(f'Generate and Download {selected_plot}')
        if export_button:
            # Save the plot as a file
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=x_variable, y=y_variable, data=df)
            plt.xlabel(x_variable)
            plt.ylabel(y_variable)
            plt.title('Box Plot')
            plt.tight_layout()
            plt.xticks(rotation=45)
            plt.savefig(f"{selected_plot}.png")
            
            # Increment the download count
            increment_download_count()

            # Provide a download link for the plot
            with open(f"{selected_plot}.png", 'rb') as file:
                b64 = base64.b64encode(file.read()).decode()  # Encode the file content in base64
                href = f'<a href="data:image/png;base64,{b64}" download="{selected_plot}.png">Click here to download the {selected_plot}</a>'
                st.markdown(href, unsafe_allow_html=True)

            st.pyplot(fig)

    elif selected_plot == 'Pie Plot':
        st.write('###### Pie Plot: ')
        st.write("###### Definition : A pie chart is a circular plot divided into sectors, where each sector represents a category or proportion of a whole.")
        st.write("###### Example: A pie chart can be used to illustrate the distribution of expenses in a monthly budget. Each sector represents a category (e.g., rent, groceries, transportation), and the size of the sector represents the proportion of the budget allocated to that category.")

        st.subheader('Pie Chart')
        variable = st.selectbox('Select variable', df.columns)

        # Download the plot
        export_button = st.button(f'Generate and Download {selected_plot}')
        if export_button:
            # Save the plot as a file
            # Check if variable is a categorical column
            if df[variable].dtype == 'object':
                value_counts = df[variable].value_counts()
                labels = value_counts.index
                counts = value_counts.values

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(counts, labels=labels)
                ax.set_title('Pie Chart')
                plt.tight_layout()
                plt.xticks(rotation=45)
                plt.savefig(f"{selected_plot}.png")
                
                # Increment the download count
                increment_download_count()

                # Provide a download link for the plot
                with open(f"{selected_plot}.png", 'rb') as file:
                    b64 = base64.b64encode(file.read()).decode()  # Encode the file content in base64
                    href = f'<a href="data:image/png;base64,{b64}" download="{selected_plot}.png">Click here to download the {selected_plot}</a>'
                    st.markdown(href, unsafe_allow_html=True)

                st.pyplot(fig)
            else:
                st.write('Selected variable is not categorical.')      
                
# Display the download count
def display_download_count(download_count: int) -> None:
    """
    Display the download count.

    This function takes the download count as input and displays it using Streamlit.

    Parameters:
        download_count (int): The current download count.

    Returns:
        None.

    """
    st.write(f"No of Plots Downloaded: {download_count}")