""" CLEAN CODE """

# Importing Necessary Libraries
import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from typing import List

import streamlit as st

# Set page title and layout
st.set_page_config(page_title='Data Analysis App', layout='wide')

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
    download_count = 133

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
            
# Upload CSV or Excel file and generate report
def upload_file() -> pd.DataFrame:
    """
    Upload a CSV or Excel file and generate a data profiling report.

    This function allows the user to upload a CSV or Excel file and generates a data profiling report using pandas-profiling.
    It provides options to display the report, export it as HTML, increment the download count, and provide a download link.

    Returns:
        pd.DataFrame: The uploaded DataFrame.
    """

    st.subheader('Upload a CSV or Excel file to generate a data profiling report.')

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
        st.subheader('Your Dataset')
        st.write(df)

    # Display the top 5 rows of the dataset
    if st.button('Show Top 5 Rows', key='show_top5_before_cleaning'):
        st.subheader('Top 5 Rows')
        st.write(df.head(5))

    # Display the bottom 5 rows of the dataset
    if st.button('Show Bottom 5 Rows', key='show_bottom5_before_cleaning'):
        st.subheader('Bottom 5 Rows')
        st.write(df.tail(5))

    # Display the shape of the dataset
    if st.button('Show Shape of the Dataset', key='show_shape_before_cleaning'):
        st.subheader('Shape of the dataset')
        st.write(f'The shape of your dataset is {df.shape[0]} rows and {df.shape[1]} columns')

    # Display the types of columns
    if st.button('Show Types of Columns', key='show_datatypes_before_cleaning'):
        st.subheader('The types of your columns')
        st.write(df.dtypes)

    # Display missing values and duplicate values
    if st.button('Show Missing Values and Duplicate Values', key='show_duplicates_before_cleaning'):
        st.subheader('Missing Values and Duplicate Values In Your Dataset')
        missing_values = df.isnull().sum()
        missing_values_formatted = ', '.join(f"{column} - {count}" for column, count in missing_values.items())
        st.write(f"The DataFrame contains missing values:\n\n{missing_values_formatted}\n")
        st.write(f'The number of duplicated rows in your dataset is {df.duplicated().sum()}')

    # Display descriptive statistics
    if st.button('Show Descriptive Statistics', key='show_stats_before_cleaning'):
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
        missing_values_action = st.selectbox('Missing Values Action', ['Do Nothing', 'Dropna', 'Fillna', 'Replace', 'Simple Imputer', 'KNN Imputer'], key='missing_values_action')
        
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
        remove_outliers = st.selectbox('Remove outliers', ['Box Plot', 'Standard Deviation', 'Z Score'])
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
        categorical_columns = st.multiselect('Categorical columns to one-hot encode', df.select_dtypes(include=['object']).columns)
        df = pd.get_dummies(df, columns=categorical_columns)
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
        st.subheader('Your Dataset')
        st.write(df)

    # Display the top 5 rows of the dataset
    if st.button('Show Top 5 Rows', key='show_top_rows'):
        st.subheader('Top 5 Rows')
        st.write(df.head(5))

    # Display the bottom 5 rows of the dataset
    if st.button('Show Bottom 5 Rows', key='show_bottom_rows'):
        st.subheader('Bottom 5 Rows')
        st.write(df.tail(5))

    # Display the shape of the dataset
    if st.button('Show Shape of the Dataset', key='show_dataset_shape'):
        st.subheader('Shape of the dataset')
        st.write(f'The shape of your dataset is {df.shape[0]} rows and {df.shape[1]} columns')

    # Display the types of columns
    if st.button('Show Types of Columns', key='show_dataset_columns'):
        st.subheader('The types of your columns')
        st.write(df.dtypes)

    # Display missing values and duplicate values
    if st.button('Show Missing Values and Duplicate Values', key='show_dataset_missing'):
        st.subheader('Missing Values and Duplicate Values In Your Dataset')
        missing_values = df.isnull().sum()
        missing_values_formatted = ', '.join(f"{column} - {count}" for column, count in missing_values.items())
        st.write(f"The DataFrame contains missing values:\n\n{missing_values_formatted}\n")
        st.write(f'The number of duplicated rows in your dataset is {df.duplicated().sum()}')

    # Display descriptive statistics
    if st.button('Show Descriptive Statistics', key='show_dataset_statistics'):
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
    plot_types = ['scatter', 'line', 'bar', 'histogram', 'box', 'pie']
    aggregation_levels = ['Yearly', 'Monthly', 'Daily']

    # Select plot type
    selected_plot = st.selectbox('Select plot type', plot_types)

    # Select aggregation level
    selected_aggregation = st.selectbox('Select aggregation level', aggregation_levels)

    if selected_plot == 'scatter':
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

        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values)
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title('Scatter Plot')
        plt.tight_layout()
        plt.xticks(rotation=45)
        if df[x_variable].dtype == 'datetime64[ns]':
            if selected_aggregation == 'Yearly':
                    plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired
            elif selected_aggregation == 'Monthly':
                    plt.xticks(x_values, x_values.strftime('%Y-%m'))  # Format x-axis ticks as desired
            elif selected_aggregation == 'Daily':
                    plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired
        st.pyplot()

    elif selected_plot == 'line':
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

        plt.figure(figsize=(12, 6))
        plt.plot(x_values, y_values)
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title('Line Plot')
        plt.tight_layout()
        plt.xticks(rotation=45)
        if df[x_variable].dtype == 'datetime64[ns]':
            if selected_aggregation == 'Yearly':
                plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired
            elif selected_aggregation == 'Monthly':
                plt.xticks(x_values, x_values.strftime('%Y-%m'))  # Format x-axis ticks as desired
            elif selected_aggregation == 'Daily':
                plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired
        st.pyplot()

    elif selected_plot == 'bar':
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
                x_values = df[x_variable]
                y_values = df[y_variable]
            else:
                # Assign numerical values to categories
                categories = df[x_variable].unique()
                category_dict = {category: i for i, category in enumerate(categories)}
                x_values = df[x_variable].map(category_dict)
                y_values = df[y_variable]

        plt.figure(figsize=(8, 6))
        plt.bar(x_values, y_values)
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title('Bar Plot')
        plt.tight_layout()
        plt.xticks(rotation=45)
        if df[x_variable].dtype == 'datetime64[ns]':
            if selected_aggregation == 'Yearly':
                plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired
            elif selected_aggregation == 'Monthly':
                plt.xticks(x_values, x_values.strftime('%Y-%m'))  # Format x-axis ticks as desired
            elif selected_aggregation == 'Daily':
                plt.xticks(x_values, x_values.astype(str))  # Format x-axis ticks as desired
        st.pyplot()

    elif selected_plot == 'histogram':
        st.subheader('Histogram')
        variable = st.selectbox('Select variable', df.columns)

        # Check if variable is a numerical column
        if np.issubdtype(df[variable].dtype, np.number):
            plt.figure(figsize=(8, 6))
            plt.hist(df[variable], bins='auto')
            plt.xlabel(variable)
            plt.ylabel('Frequency')
            plt.title('Histogram')
            plt.tight_layout()
            plt.xticks(rotation=45)
            st.pyplot()
        else:
            st.write('Selected variable is not numerical.')

    elif selected_plot == 'box':
        st.subheader('Box Plot')
        x_variable = st.selectbox('Select x-axis variable', df.columns)
        y_variable = st.selectbox('Select y-axis variable', df.columns)

        plt.figure(figsize=(8, 6))
        sns.boxplot(x=x_variable, y=y_variable, data=df)
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title('Box Plot')
        plt.tight_layout()
        plt.xticks(rotation=45)
        st.pyplot()

    elif selected_plot == 'pie':
        st.subheader('Pie Chart')
        variable = st.selectbox('Select variable', df.columns)

        # Check if variable is a categorical column
        if df[variable].dtype == 'object':
            plt.figure(figsize=(8, 6))
            plt.pie(df[variable].value_counts(), labels=df[variable].unique())
            plt.title('Pie Chart')
            plt.tight_layout()
            plt.xticks(rotation=45)
            st.pyplot()
        else:
            st.write('Selected variable is not categorical.')       

def generate_report(df):
    """
    Generate a profiling report for the DataFrame and provide interactivity options for exporting the report.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        None.
    """

    st.subheader('Do you need to generate the report?')
    submit_button = st.button('Yes')
    
    # Generate the pandas profiling report
    profile = ProfileReport(df, 
                            title="Profiling Report", 
                            explorative = True, 
                            dark_mode = True,
                            dataset={
                            "description": "This app is created by - Pradeepchandra Reddy S C (a.k.a soopertramp07)",
                            "copyright_holder": "soopertramp07",
                            "copyright_year": "2023",
                            "url": "https://www.linkedin.com/in/pradeepchandra-reddy-s-c/"})     

    if submit_button:
        # Display the profiling report using pandas_profiling
        st_profile_report(profile)

    st.subheader('Export Report')
    export_button = st.button('Export Report as HTML')

    if export_button:
        profile.to_file("profiling_report.html")
        st.success("Report exported successfully as HTML! ✅")

        # Increment the download count
        increment_download_count()

        # Provide a download link for the exported report
        with open("profiling_report.html", 'rb') as file:
            b64 = base64.b64encode(file.read()).decode()  # Encode the file content in base64
            href = f'<a href="data:text/html;base64,{b64}" download="data_analysis_report.html">Download The Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.write(":arrow_up: Click Above To Download The Report, Thank You! :pray:")

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
    st.write(f"Downloaders Count: {download_count}")

# Main app
def data_analysis_app() -> None:
    """
    Run the Data Analysis App.

    This function runs the data analysis app, including uploading and generating a report, and displaying the download count.

    Returns:
        None.

    """

    st.title('Data Analysis App :chart_with_upwards_trend:')
    
    download_count = initialize_download_count()

    df = upload_file()
    basic_investigation(df)
    cleaned_df = apply_data_cleaning(df)
    display_after_cleaning(cleaned_df)
    generate_download_link(cleaned_df, 'cleaned_dataset')
    plot_data(cleaned_df)
        
    generate_report(cleaned_df)
    
    # Display the download count
    display_download_count(download_count)

if __name__ == '__main__':
    data_analysis_app()
