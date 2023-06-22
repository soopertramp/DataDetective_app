""" CLEAN CODE """

# Importing Necessary Libraries

import streamlit as st
from utils import (initialize_download_count,
                   increment_download_count, 
                   upload_file,
                   basic_investigation,
                   apply_data_cleaning,
                   display_after_cleaning,
                   generate_download_link,
                   plot_data,
                   generate_report,
                   display_download_count)

# Set page title and layout
st.set_page_config(page_title='Data Analysis App', layout='wide')

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
    download_count = increment_download_count()

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