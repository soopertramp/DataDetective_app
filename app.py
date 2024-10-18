# Importing Necessary Libraries
from utils import (apply_data_cleaning, basic_investigation,
                   display_after_cleaning, display_download_count,
                   generate_download_link, increment_download_count,
                   initialize_download_count, plot_data, upload_file)

import streamlit as st

# Set page title and layout 
st.set_page_config(page_title="Analyst", layout='wide')

# Main app
def data_analysis_app(): 
    """
    Run the Data Analysis App.

    This function runs the data analysis app, including uploading and generating a report, and displaying the download count.
    """
    st.title("Hello, I'm A DataDetective :cop:")
    st.write(" I'm here to help you analyze your data. Upload a CSV or Excel file, and I'll provide you with insights and investigation before diving into the data. Let's get started on your investigation and uncover valuable information from your data!")
    
    # Initialize download count
    download_count = initialize_download_count()

    # Upload file and perform basic investigation
    df = upload_file()
    basic_investigation(df)

    # Apply data cleaning and display cleaned data
    cleaned_df = apply_data_cleaning(df)
    display_after_cleaning(cleaned_df)

    # Generate download link for cleaned dataset
    generate_download_link(cleaned_df, 'cleaned_dataset')

    # Plot data
    plot_data(cleaned_df)

    # Increment download count
    download_count = increment_download_count()

    # Display the download count
    display_download_count(download_count)  

if __name__ == '__main__':
    data_analysis_app()