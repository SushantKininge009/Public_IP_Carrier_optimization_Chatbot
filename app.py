#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
import io
import re
from fuzzywuzzy import process
from google.cloud import storage
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# --- Configuration ---
PROJECT_ID = "gen-ai-rajan-labs"
LOCATION = "us-central1"
GCS_BUCKET_NAME = "publicip_carrier_data"

# File paths in GCS
PEAK_USAGE_FILE = "carrier_peak_usage.xlsx"
ACCOUNT_MANAGERS_FILE = "carrier_account_managers.xlsx"
SUPPORT_FILE = "carrier_first_line_support.xlsx"

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-1.5-pro-002")

# Initialize GCS Client
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(GCS_BUCKET_NAME)

@st.cache_data
def load_data():
    """Loads the dataframes from GCS and merges them."""
    try:
        def load_excel_from_gcs(blob_name):
            """Downloads an Excel file from GCS and loads it into a pandas DataFrame."""
            try:
                blob = bucket.blob(blob_name)
                content = blob.download_as_bytes()
                df = pd.read_excel(io.BytesIO(content))
                print(f"✅ Successfully loaded: {blob_name}")
                return df
            except Exception as e:
                print(f"❌ Error loading {blob_name}: {str(e)}")
                return None

        df_usage = load_excel_from_gcs(PEAK_USAGE_FILE)
        df_managers = load_excel_from_gcs(ACCOUNT_MANAGERS_FILE)
        df_support = load_excel_from_gcs(SUPPORT_FILE)

        # --- Clean Column Names ---
        def clean_column_names(df):
            if df is None:
                return None
            df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '', regex=True)
            return df

        df_usage = clean_column_names(df_usage)
        print("\nColumns in df_usage AFTER cleaning:")
        print(df_usage.columns)

        df_managers = clean_column_names(df_managers)
        df_support = clean_column_names(df_support)

        # --- Standardize Carrier Names ---
        def standardize_carrier_name(name):
            if pd.isna(name):
                return None
            name = str(name).strip().upper()
            variations = {
                'GT IRELAND': 'GT_IRELAN',
                'BLARO_I': 'BLARO_DR',
                'BLARO PERU': 'BLARO_PE',
                'BLARO ARGENTINA': 'BLARO_AR',
                'BLARO CHILE': 'BLARO',
                'BLARO BRAZIL': 'BLARO BR',
                'BLARO_COLOMBIA': 'BLARO_CO',
                'FANX_TELECOM': 'FANX TELE',
                'GAROC_TELECOM': 'GAROC TELE',
                'KUBAI TELECOM': 'KUBAI TEL',
                'TEL_ARABIA': 'TEL ARABIA',
                'EL_EGIPT': 'EL EGIPT',
                'RANGE_MEDITEL': 'RANGE MEDITEL',
                'LOTSWANA_TEL': 'LOTSWANA_TEL'
            }
            return variations.get(name, name)

        if df_usage is not None and 'carrier_name' in df_usage.columns:
            df_usage['standardized_carrier_name'] = df_usage['carrier_name'].apply(standardize_carrier_name)
        if df_managers is not None and 'carrier_name' in df_managers.columns:
            df_managers['standardized_carrier_name'] = df_managers['carrier_name'].apply(standardize_carrier_name)
        if df_support is not None and 'carrier_name' in df_support.columns:
            df_support['standardized_carrier_name'] = df_support['carrier_name'].apply(standardize_carrier_name)

        # --- Merge the DataFrames ---
        df_merged = None
        if df_usage is not None and 'standardized_carrier_name' in df_usage.columns and 'peak_usage' in df_usage.columns and 'configured_capacity' in df_usage.columns:
            df_merged = df_usage

            if df_managers is not None and 'standardized_carrier_name' in df_managers.columns:
                manager_cols = ['standardized_carrier_name', 'your_company_account_manager_name', 'your_company_account_manager_email', 'carrier_company_account_manager_name', 'carrier_company_account_manager_email']
                df_managers_subset = df_managers[manager_cols].drop_duplicates(subset=['standardized_carrier_name'])
                df_merged = pd.merge(df_merged, df_managers_subset, on='standardized_carrier_name', how='left')
                print("Merged Manager data.")

            if df_support is not None and 'standardized_carrier_name' in df_support.columns:
                support_cols = ['standardized_carrier_name', 'first_line_contact_name', 'first_line_contact_email']
                df_support_subset = df_support[support_cols].drop_duplicates(subset=['standardized_carrier_name'])
                df_merged = pd.merge(df_merged, df_support_subset, on='standardized_carrier_name', how='left')
                print("Merged Support data.")
            print("✅ Data loaded and merged successfully.")
            return df_merged
        else:
            print("❌ Could not load and merge data due to missing usage data or required columns.")
            return None
    except Exception as e:
        print(f"❌ An error occurred during data loading and merging: {e}")
        return None
df_merged = load_data()

st.title("Public IP Carrier Analysis Chatbot")

if df_merged is not None:
    st.subheader("Underutilized Carriers")
    if 'standardized_carrier_name' in df_merged.columns and 'configured_capacity' in df_merged.columns and 'peak_usage' in df_merged.columns and 'usage_percentage' in df_merged.columns and 'proposed_capacity' in df_merged.columns:
        underutilized_threshold = st.sidebar.slider("Usage Threshold (%)", 0.0, 100.0, 40.0)
        df_underutilized = df_merged.copy()
        df_underutilized['peak_usage'] = pd.to_numeric(df_underutilized['peak_usage'], errors='coerce')
        df_underutilized['configured_capacity'] = pd.to_numeric(df_underutilized['configured_capacity'], errors='coerce')
        df_underutilized.dropna(subset=['peak_usage', 'configured_capacity'], inplace=True)
        df_underutilized = df_underutilized[df_underutilized['configured_capacity'] > 0]
        df_underutilized['usage_percentage'] = (df_underutilized['peak_usage'] / df_underutilized['configured_capacity']) * 100
        df_underutilized = df_underutilized[df_underutilized['usage_percentage'] < underutilized_threshold].copy()
        df_underutilized['proposed_capacity'] = (df_underutilized['configured_capacity'] * 0.5).round().astype(int)
        st.dataframe(df_underutilized[['standardized_carrier_name', 'configured_capacity', 'peak_usage', 'usage_percentage', 'proposed_capacity']])
    else:
        st.warning("Required columns for underutilized carriers not found.")

    st.subheader("Ask a Question about the Data")
    user_question = st.text_input("Your Question:")
    ask_button = st.button("Ask")

    if ask_button:
        if user_question:
            with st.spinner("Thinking..."):
                context_prompt = f"""
                    You are a helpful AI assistant for a telecom solutions architect. You have access to data about Public IP Carriers.
                    The data includes carrier names, peak SIP session usage, configured capacity, account manager details (both internal and carrier-side),
                    and first-line support contacts. The data comes from three sources and has been merged. Carrier names have been standardized.
                    
                    You are a helpful AI assistant for a telecom solutions architect. You have access to data about Public IP Carriers.
                    The data includes the following information for each carrier:
                    - **carrier_name**: The name of the carrier.
                    - **peak_usage**: The peak number of concurrent SIP sessions observed.
                    - **configured_capacity**: The total configured capacity for SIP sessions (this represents the number of sessions).
                    - **configured_trunks**: The number of configured trunks (this is a separate metric from session capacity).
                    - **used_trunks**: The number of trunks that were used.
                    - Account manager details (both internal and carrier-side).
                    - First-line support contacts.


                    Available Data Columns Overview: {', '.join(df_merged.columns.tolist()) if df_merged is not None else 'Data not available'}
                    Total Carriers in Merged Data: {len(df_merged['standardized_carrier_name'].unique()) if df_merged is not None else 0}

                    **Full Carrier Data:**
                    {df_merged.to_string()}

                    User Query: {user_question}

                    Answer:
                    """
                try:
                    response = model.generate_content(context_prompt)
                    st.write("Answer:", response.text)
                except Exception as e:
                    st.error(f"Error calling Vertex AI Model: {e}")
        else:
            st.warning("Please enter your question.")
else:
    st.error("Data could not be loaded. Please check the backend setup.")


# In[ ]:




