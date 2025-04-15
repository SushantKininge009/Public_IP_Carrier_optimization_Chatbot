#!/usr/bin/env python
# coding: utf-8

# In[12]:


pip install pandas google-cloud-storage google-cloud-aiplatform gcsfs fsspec openpyxl streamlit fuzzywuzzy[speedup] python-Levenshtein langchain langchain-google-vertexai db-dtypes # db-dtypes sometimes needed by pandas agents


# In[13]:


import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)


# In[1]:


# Install necessary libraries (if not already on the Vertex AI Workbench instance)
# %pip install pandas openpyxl google-cloud-storage google-cloud-aiplatform fuzzywuzzy[speedup]

import pandas as pd
import io
import re # For potential regex cleaning
from fuzzywuzzy import process # For fuzzy name matching
from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
PROJECT_ID = "gen-ai-rajan-labs"  # Replace with your Project ID
LOCATION = "us-central1"  # e.g., "us-central1"
GCS_BUCKET_NAME = "publicip_carrier_data" # Replace with your bucket name
GCS_BUCKET_NAME = "publicip_carrier_data" 

# File paths in GCS
PEAK_USAGE_FILE = "carrier_peak_usage.xlsx"
ACCOUNT_MANAGERS_FILE = "carrier_account_managers.xlsx"
SUPPORT_FILE = "carrier_first_line_support.xlsx"

# Analysis Parameters
USAGE_THRESHOLD_PERCENT = 40.0
CAPACITY_REDUCTION_FACTOR = 0.5

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load the Gemini model
model = GenerativeModel("gemini-1.5-pro-002") # Or choose a newer/different version if needed

# Initialize GCS Client
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(GCS_BUCKET_NAME)
print("Setup Complete. Vertex AI SDK and GCS Client Initialized.")


# In[2]:


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

# Load the data
df_usage = load_excel_from_gcs(PEAK_USAGE_FILE)
df_managers = load_excel_from_gcs(ACCOUNT_MANAGERS_FILE)
df_support = load_excel_from_gcs(SUPPORT_FILE)

# Display data (if loaded)
if df_usage is not None:
    print("\nPeak Usage Data Sample:")
    print(df_usage.head())
if df_managers is not None:
    print("\nAccount Managers Data Sample:")
    print(df_managers.head())
if df_support is not None:
    print("\nSupport Data Sample:")
    print(df_support.head())


# In[3]:


import pandas as pd
# Load your data (assuming you've already loaded these)
#df_usage = pd.read_excel('carrier_peak_usage.xlsx')
#df_managers = pd.read_excel('carrier_account_managers.xlsx')
#df_support = pd.read_excel('carrier_first_line_support.xlsx')

# Standardize carrier names across all datasets
def standardize_carrier_name(name):
    if pd.isna(name):
        return None
    name = str(name).strip().upper()
    # Handle known variations
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

# Apply standardization to each DataFrame using the correct column names
if 'carrier_name_' in df_usage.columns:
    df_usage['standardized_carrier_name'] = df_usage['carrier_name_'].apply(standardize_carrier_name)
else:
    df_usage['standardized_carrier_name'] = None

if 'carrier_name' in df_managers.columns:
    df_managers['standardized_carrier_name'] = df_managers['carrier_name'].apply(standardize_carrier_name)
else:
    df_managers['standardized_carrier_name'] = None

if 'carrier_name_' in df_support.columns:
    df_support['standardized_carrier_name'] = df_support['carrier_name_'].apply(standardize_carrier_name)
else:
    df_support['standardized_carrier_name'] = None

# Display results for verification
print("\nUsage Data with Standardized Names:")
if 'carrier_name_' in df_usage.columns and 'standardized_carrier_name' in df_usage.columns:
    print(df_usage[['carrier_name_', 'standardized_carrier_name']].head())
else:
    print("Could not display - columns not found")

print("\nManagers Data with Standardized Names:")
if 'carrier_name' in df_managers.columns and 'standardized_carrier_name' in df_managers.columns:
    print(df_managers[['carrier_name', 'standardized_carrier_name']].head())
else:
    print("Could not display - columns not found")

print("\nSupport Data with Standardized Names:")
if 'carrier_name_' in df_support.columns and 'standardized_carrier_name' in df_support.columns:
    print(df_support[['carrier_name_', 'standardized_carrier_name']].head())
else:
    print("Could not display - columns not found")


# In[4]:


def clean_column_names(df):
    """Standardizes column names (lowercase, replace spaces with underscores)."""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '', regex=True)
    return df

def standardize_carrier_names(df_target, df_reference, target_col, reference_col, threshold=85):
    """
    Standardizes carrier names in df_target based on names in df_reference
    using fuzzy matching. Adds a 'standardized_carrier_name' column.
    """
    if df_target is None or df_reference is None:
        print("One or both dataframes are None, skipping standardization.")
        return df_target

    reference_names = df_reference[reference_col].unique().tolist()
    standardized_names = {}

    for name in df_target[target_col].unique():
        if pd.isna(name):
            standardized_names[name] = None
            continue
        # Find the best match above the threshold
        match, score = process.extractOne(str(name), reference_names)
        if score >= threshold:
            standardized_names[name] = match
            # print(f"Matched '{name}' to '{match}' with score {score}") # Debugging
        else:
            standardized_names[name] = str(name) # Keep original if no good match
            # print(f"No good match for '{name}' (best: '{match}', score: {score}). Keeping original.") # Debugging

    df_target['standardized_carrier_name'] = df_target[target_col].map(standardized_names)
    return df_target

# --- Apply Cleaning ---
if df_usage is not None:
    df_usage = clean_column_names(df_usage)
    # Assume 'carrier_name' is the column in df_usage
    # Create a reference name list (e.g., from the managers file, assuming it's cleaner or more complete)
    if df_managers is not None:
      df_managers = clean_column_names(df_managers)
      # Assume the column is 'carrier_name' in managers sheet
      reference_carrier_names_list = df_managers['carrier_name'].dropna().unique().tolist()

      # Standardize usage df based on manager df names
      standardized_names_map_usage = {}
      usage_carrier_col = 'carrier_name' # Adjust if column name is different after cleaning
      if usage_carrier_col in df_usage.columns:
          for name in df_usage[usage_carrier_col].unique():
                if pd.isna(name): continue
                match, score = process.extractOne(str(name), reference_carrier_names_list)
                if score >= 85: # Adjust threshold as needed
                    standardized_names_map_usage[name] = match
                else:
                    standardized_names_map_usage[name] = str(name) # Keep original if no good match
          df_usage['standardized_carrier_name'] = df_usage[usage_carrier_col].map(standardized_names_map_usage)
          print("Standardized carrier names in Usage data.")
      else:
          print(f"Column '{usage_carrier_col}' not found in usage dataframe after cleaning.")
          # Handle error or assign a default standardized name column
          df_usage['standardized_carrier_name'] = df_usage[usage_carrier_col] if usage_carrier_col in df_usage.columns else None

    else: # If manager df is not available, standardize based on its own names (less ideal)
       print("Manager data not loaded. Standardizing Usage data based on its own unique names.")
       df_usage['standardized_carrier_name'] = df_usage['carrier_name'] # Or apply self-referential fuzzy matching if needed


# Standardize managers and support dfs (use one as the primary reference or self-reference)
if df_managers is not None:
   # Standardize against itself or a master list if you have one
   df_managers['standardized_carrier_name'] = df_managers['carrier_name'] # Simplest approach: assume names are already the reference standard
   print("Assigned standardized names in Managers data.")

if df_support is not None:
    df_support = clean_column_names(df_support)
    support_carrier_col = 'carrier_name' # Adjust if needed
    if support_carrier_col in df_support.columns:
      if df_managers is not None: # Use managers list as reference if available
          standardized_names_map_support = {}
          for name in df_support[support_carrier_col].unique():
                if pd.isna(name): continue
                match, score = process.extractOne(str(name), reference_carrier_names_list)
                if score >= 85:
                    standardized_names_map_support[name] = match
                else:
                    standardized_names_map_support[name] = str(name)
          df_support['standardized_carrier_name'] = df_support[support_carrier_col].map(standardized_names_map_support)
          print("Standardized carrier names in Support data based on Managers list.")

      else: # Fallback to self-reference
          df_support['standardized_carrier_name'] = df_support[support_carrier_col]
          print("Standardized carrier names in Support data based on its own names.")
    else:
       print(f"Column '{support_carrier_col}' not found in support dataframe after cleaning.")
       df_support['standardized_carrier_name'] = None


# Display standardized names (optional check)
if df_usage is not None: print("\nUsage Data with Standardized Names:\n", df_usage[['carrier_name', 'standardized_carrier_name']].head())
# ... similar checks for df_managers and df_support


# In[8]:


print("Usage columns:", df_usage.columns.tolist())
print("Managers columns:", df_managers.columns.tolist())
print("Support columns:", df_support.columns.tolist())


# In[5]:


# Merge the dataframes using the standardized carrier name
df_merged = None
if df_usage is not None and 'standardized_carrier_name' in df_usage.columns:
    df_merged = df_usage

    if df_managers is not None and 'standardized_carrier_name' in df_managers.columns:
        # Select only necessary manager columns and rename before merge to avoid conflicts if needed
        manager_cols = ['standardized_carrier_name', 'your_company_account_manager_name', 'your_company_account_manager_email', 'carrier_company_account_manager_name', 'carrier_company_account_manager_email']
        df_managers_subset = df_managers[manager_cols].drop_duplicates(subset=['standardized_carrier_name'])
        df_merged = pd.merge(df_merged, df_managers_subset, on='standardized_carrier_name', how='left')
        print("Merged Manager data.")

    if df_support is not None and 'standardized_carrier_name' in df_support.columns:
         # Select support columns and rename before merge
        support_cols = ['standardized_carrier_name', 'first_line_contact_name', 'first_line_contact_email']
        df_support_subset = df_support[support_cols].drop_duplicates(subset=['standardized_carrier_name'])
        df_merged = pd.merge(df_merged, df_support_subset, on='standardized_carrier_name', how='left')
        print("Merged Support data.")

else:
    print("Cannot merge data as Usage data or standardized name column is missing.")

if df_merged is not None:
    print("\nMerged Data Head:\n", df_merged.head())
    # Handle potential NaN values introduced by merging (optional: fill with defaults like 'N/A')
    # df_merged = df_merged.fillna('N/A')
else:
    print("Merging failed or was skipped.")
    


# In[6]:


df_analysis = None
if df_merged is not None and 'peak_usage' in df_merged.columns and 'configured_capacity' in df_merged.columns:
    # Make a copy for analysis to avoid modifying the merged view directly
    df_analysis = df_merged.copy()

    # Ensure numeric types and handle potential errors/zeros
    df_analysis['peak_usage'] = pd.to_numeric(df_analysis['peak_usage'], errors='coerce')
    df_analysis['configured_capacity'] = pd.to_numeric(df_analysis['configured_capacity'], errors='coerce')
    df_analysis.dropna(subset=['peak_usage', 'configured_capacity'], inplace=True) # Drop rows where conversion failed
    df_analysis = df_analysis[df_analysis['configured_capacity'] > 0] # Avoid division by zero

    # Calculate Usage Percentage
    df_analysis['usage_percentage'] = (df_analysis['peak_usage'] / df_analysis['configured_capacity']) * 100

    # Identify Underutilized Carriers
    df_underutilized = df_analysis[df_analysis['usage_percentage'] < USAGE_THRESHOLD_PERCENT].copy()

    # Calculate Proposed New Capacity
    df_underutilized['proposed_capacity'] = (df_underutilized['configured_capacity'] * CAPACITY_REDUCTION_FACTOR).round().astype(int) # Round to nearest int

    print(f"\nIdentified {len(df_underutilized)} underutilized carriers (Usage < {USAGE_THRESHOLD_PERCENT}%).")
    if not df_underutilized.empty:
        print("Underutilized Carriers Sample:\n", df_underutilized[['standardized_carrier_name', 'configured_capacity', 'peak_usage', 'usage_percentage', 'proposed_capacity']].head())
else:
    print("Cannot perform analysis. Merged data or required columns ('peak_usage', 'configured_capacity') are missing.")


# In[7]:


def ask_chatbot(query):
    """Sends a query with context to the LLM and returns the answer."""
    if df_merged is None:
        return "Sorry, the data is not loaded or merged correctly. I cannot answer questions yet."

    # --- Basic Context Preparation ---
    # For more complex queries, you might need to dynamically select relevant data
    # For now, provide a general overview and let the LLM figure it out.
    # You could enhance this to find specific carrier data if the query mentions one.

    context_prompt = f"""
    You are a helpful AI assistant for a telecom solutions architect. You have access to data about Public IP Carriers.
    The data includes carrier names, peak SIP session usage, configured capacity, account manager details (both internal and carrier-side), and first-line support contacts.
    The data comes from three sources and has been merged. Carrier names have been standardized.

    Use the provided data snapshot (if any) and your general knowledge to answer the user's query accurately and professionally.

    Available Data Columns Overview: {', '.join(df_merged.columns.tolist()) if df_merged is not None else 'Data not available'}
    Total Carriers in Merged Data: {len(df_merged['standardized_carrier_name'].unique()) if df_merged is not None else 0}

    User Query: {query}

    Answer:
    """

    # --- Optional: Add Specific Data Snippet if Query is Specific ---
    # Example: if query mentions a carrier name, find its row(s) and add to context
    # carrier_match = re.search(r'carrier\s+([A-Za-z0-9_\-\s]+)', query, re.IGNORECASE)
    # if carrier_match:
    #    carrier_name_query = carrier_match.group(1).strip()
    #    # Try finding the carrier using standardized name
    #    carrier_data = df_merged[df_merged['standardized_carrier_name'].str.contains(carrier_name_query, case=False, na=False)]
    #    if not carrier_data.empty:
    #       context_prompt += f"\n\nRelevant Data for '{carrier_name_query}':\n{carrier_data.to_string()}"


    try:
        response = model.generate_content(context_prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Vertex AI Model: {e}")
        return "Sorry, I encountered an error trying to process your request with the AI model."

# --- Example Usage (in a notebook cell) ---
# user_question = "Tell me about the peak usage for BLARO ARGENTINA."
# answer = ask_chatbot(user_question)
# print(answer)

# user_question = "Who is the account manager from our company for Carrier XYZ?"
# answer = ask_chatbot(user_question)
# print(answer)

# user_question = "How many carriers are there in total?"
# answer = ask_chatbot(user_question)
# print(answer)


# In[15]:


def generate_capacity_reduction_email(carrier_info):
    """Generates an email notification using the LLM based on carrier data."""

    # Ensure all needed info is present, handle potential missing data gracefully
    carrier_name = carrier_info.get('standardized_carrier_name', 'N/A')
    current_capacity = carrier_info.get('configured_capacity', 'N/A')
    peak_usage = carrier_info.get('peak_usage', 'N/A')
    usage_percent = carrier_info.get('usage_percentage', 'N/A')
    new_capacity = carrier_info.get('proposed_capacity', 'N/A')

    your_am_name = carrier_info.get('your_company_account_manager_name', 'Your Account Manager')
    your_am_email = carrier_info.get('your_company_account_manager_email', 'your_am@yourcompany.com') # Provide a default or fetch dynamically
    carrier_am_name = carrier_info.get('carrier_company_account_manager_name', 'Carrier Contact')
    carrier_am_email = carrier_info.get('carrier_company_account_manager_email', '') # Email is crucial
    support_name = carrier_info.get('first_line_contact_name', 'Carrier Support')
    support_email = carrier_info.get('first_line_contact_email', '') # Email is crucial


    # Construct the prompt for the LLM
    email_prompt = f"""
    Generate a professional email notification regarding a planned capacity reduction for a Public IP Carrier voice trunk.

    **Instructions:**
    1.  Be polite and professional.
    2.  Clearly state the reason for the reduction (peak usage consistently below {USAGE_THRESHOLD_PERCENT}% of configured capacity).
    3.  Mention the current configured capacity, the observed peak usage (and percentage), and the proposed new capacity (which is 50% of the current).
    4.  Address the email primarily to the Carrier Company Account Manager.
    5.  CC the Account Manager from our company and the Carrier's First Line Support contact.
    6.  Provide contact information (Our Company's AM) for questions or discussion.
    7.  Suggest a timeframe for discussion before the change is implemented (e.g., "within the next two weeks").

    **Carrier Details:**
    * Carrier Name: {carrier_name}
    * Current Configured Capacity: {current_capacity} sessions
    * Observed Peak Usage: {peak_usage:.0f} sessions ({usage_percent:.1f}%)
    * Proposed New Capacity: {new_capacity} sessions
    * Carrier Account Manager: {carrier_am_name} ({carrier_am_email})
    * Our Account Manager: {your_am_name} ({your_am_email})
    * Carrier First Line Support: {support_name} ({support_email})

    **Generate the email with a clear Subject line and Body.**
    """

    # Check if essential email addresses are present
    if not carrier_am_email:
        return {"error": f"Missing Carrier Account Manager email for {carrier_name}. Cannot generate email."}

    try:
        response = model.generate_content(email_prompt)

        # Basic parsing attempt (assuming Subject: ... Body: ...)
        email_text = response.text
        subject = f"Planned Capacity Adjustment for {carrier_name} Voice Trunk" # Default subject
        body = email_text

        # Try to extract Subject if model provides it explicitly
        subject_match = re.search(r"Subject:\s*(.*)", email_text, re.IGNORECASE)
        if subject_match:
            subject = subject_match.group(1).strip()
            # Remove subject line from body if found
            body = re.sub(r"Subject:\s*.*\n?", "", body, flags=re.IGNORECASE).strip()
            body = re.sub(r"Body:\s*\n?", "", body, flags=re.IGNORECASE).strip() # Remove Body: tag if present


        cc_emails = [e for e in [your_am_email, support_email] if pd.notna(e) and '@' in str(e)] # Filter valid emails

        return {
            "to": carrier_am_email,
            "cc": cc_emails,
            "subject": subject,
            "body": body,
            "carrier": carrier_name
        }
    except Exception as e:
        print(f"Error calling Vertex AI Model for email generation: {e}")
        return {"error": f"LLM error generating email for {carrier_name}."}


# --- Example Usage (in a notebook cell) ---
# if df_underutilized is not None and not df_underutilized.empty:
#     # Generate for the first underutilized carrier
#     first_carrier_info = df_underutilized.iloc[0].to_dict()
#     generated_email = generate_capacity_reduction_email(first_carrier_info)
#     if "error" in generated_email:
#         print(generated_email["error"])
#     else:
#         print(f"--- Generated Email for {generated_email['carrier']} ---")
#         print(f"To: {generated_email['to']}")
#         print(f"CC: {', '.join(generated_email['cc'])}")
#         print(f"Subject: {generated_email['subject']}")
#         print("\nBody:\n", generated_email['body'])
# else:
#      print("No underutilized carriers found to generate emails for.")

# --- Generate for ALL underutilized carriers ---
# all_generated_emails = []
# if df_underutilized is not None and not df_underutilized.empty:
#    print("\n--- Generating Emails for All Underutilized Carriers ---")
#    for index, row in df_underutilized.iterrows():
#        carrier_info = row.to_dict()
#        email_data = generate_capacity_reduction_email(carrier_info)
#        all_generated_emails.append(email_data)
#        if "error" in email_data:
#             print(f"Failed for {carrier_info.get('standardized_carrier_name', 'Unknown')}: {email_data['error']}")
#        else:
#             print(f"Successfully generated email draft for {email_data['carrier']}")
#    # Now you have a list 'all_generated_emails' containing dicts for each email (or errors)
#    # You can review them before sending.
# else:
#    print("No underutilized carriers found.")


# In[16]:


# --- Example using smtplib (Requires SMTP server access & credentials) ---
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from google.cloud import secretmanager # Recommended for credentials

# def get_secret(secret_id, version_id="latest"):
#     """Retrieves a secret from Google Secret Manager."""
#     client = secretmanager.SecretManagerServiceClient()
#     name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"
#     response = client.access_secret_version(request={"name": name})
#     return response.payload.data.decode("UTF-8")

# def send_email_smtp(to_email, cc_emails, subject, body):
#     # --- Retrieve Credentials Securely ---
#     # SMTP_SERVER = "smtp.yourprovider.com"
#     # SMTP_PORT = 587 # Or 465 for SSL
#     # SENDER_EMAIL = get_secret("your-sender-email-secret-id") # Store in Secret Manager
#     # SENDER_PASSWORD = get_secret("your-sender-password-secret-id") # Store in Secret Manager

#     try:
#         msg = MIMEMultipart()
#         msg['From'] = SENDER_EMAIL
#         msg['To'] = to_email
#         msg['Cc'] = ", ".join(cc_emails)
#         msg['Subject'] = subject
#         msg.attach(MIMEText(body, 'plain'))

#         server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#         server.starttls() # Use TLS
#         server.login(SENDER_EMAIL, SENDER_PASSWORD)
#         recipients = [to_email] + cc_emails
#         text = msg.as_string()
#         server.sendmail(SENDER_EMAIL, recipients, text)
#         server.quit()
#         print(f"Email sent successfully to {to_email}")
#         return True
#     except Exception as e:
#         print(f"Error sending email to {to_email}: {e}")
#         return False

# # --- Usage after generating emails ---
# for email_data in all_generated_emails:
#    if "error" not in email_data:
#        print(f"\nAttempting to send email for {email_data['carrier']}...")
#        # UNCOMMENT BELOW TO ACTUALLY SEND - USE WITH CAUTION
#        # send_email_smtp(email_data['to'], email_data['cc'], email_data['subject'], email_data['body'])
#        # print("--- Email sending commented out for safety ---")
#        pass # Keep sending commented out initially
#    else:
#        print(f"Skipping send for {email_data.get('carrier', 'Unknown')} due to generation error.")


# In[ ]:




