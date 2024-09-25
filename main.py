from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import json

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
origins = [
    "https://mowenista.github.io",  # GitHub Pages URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key from environment variable
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Define request and response models
class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: str
    vega_spec: dict = None  # Optional, for Vega-Lite specifications

class Spec(BaseModel):
  spec: str


# Directory for uploaded CSV files
UPLOAD_DIRECTORY = "uploads"

# Helper function to check if a CSV file is uploaded
def is_csv_uploaded():
    return os.path.exists(UPLOAD_DIRECTORY) and os.listdir(UPLOAD_DIRECTORY)

# Helper function to clear the uploads directory
def clear_upload_directory():
    if os.path.exists(UPLOAD_DIRECTORY):
        for existing_file in os.listdir(UPLOAD_DIRECTORY):
            file_path = os.path.join(UPLOAD_DIRECTORY, existing_file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

# Root endpoint - Clears upload directory on first visit
@app.get("/")
async def read_root():
    clear_upload_directory()
    return FileResponse('static/index.html')

def generate_chart(query, df):

    prompt = f'''
    Dataset overview (top five rows): {df.head().to_markdown()}

    User Query: {query}

    Given the dataset above, generate a vega-lite specification for the user query. The data field will be inserted dynamically, so leave it empty.
    Assume the dataset overview does not represent the extent of the range.

    As much as possible avoid creating charts with unique values on the axis, as it is challenging to fit the large data sets into a compact chart.
    For example things such as titles/models/names often are hard to represent on a single axis
    
    Use other things such as scatterplots and table-based plots and bar charts. Pick the type of chart that best fits the query.
    Please ensure the chart has a compact layout, potentially combining x-axis labels and limiting horizontal stretching. 
    Also, set the width to 600 pixels for better visualization.

    Please specify the exact data columns and any filters required to adjust the pandas dataframe before adding to the chart
    '''
    response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
      {"role": "user", "content": prompt}
    ],
    response_format=Spec
    )

    return response.choices[0].message.parsed.spec

def get_feedback(query, df, spec):
  prompt = f'''
    Dataset overview (top five rows): {df.head().to_markdown()}

    User query: {query}.

    Generated Vega-lite spec: {spec}

    Please provide feedback on the generated chart whether the spec is valid in syntax and faithful to the user query.

    Assuming the data set is quite large, provide some feedback on the sizing and formatting of the chart.
    The chart should fit in a 800x800px square and not require scrolling
  '''
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {"role": "user", "content": prompt}
    ]
  )
  feedback = response.choices[0].message.content
  return feedback

def improve_response(query, df, spec, feedback):
    prompt = f'''
      Dataset overview (top five rows): {df.head().to_markdown()}

      User query: {query}.

      Generated Vega-lite spec: {spec}

      Feedback: {feedback}

      Improve the vega-lite spec with the feedback if only necessary. Otherwise, return the original spec.

    '''
    response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
      {"role": "user", "content": prompt}
    ],
      response_format=Spec
    )
    return response.choices[0].message.parsed.spec


class RelevanceCheckResponse(BaseModel):
    related: str  # "yes" or "no"
    message: str  # Explanation message


def check_query_relevance(query: str, df: pd.DataFrame):
    prompt = f'''
    I have a dataset with the following details (top five rows):
    {df.columns}

    User Query: {query}

    Determine if the query is related to the dataset. Only reject if there is a complete difference between the types of data and the query
    Respond with JSON format:
    {{
        "related": "yes" or "no",
        "message": "Your explanation to the user here."
    }}
    '''
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format=RelevanceCheckResponse
    )
    
    return json.loads(response.choices[0].message.content)

def generate_chart_description(query, spec):
    prompt = f'''
    My Query: {query}

    Vega-Lite Chart Specification: {spec}

    Please write one or two sentences about what this chart is as a response to the my query. 
    This is only a specification for the table the data is added in afterwards
    '''
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    description = response.choices[0].message.content
    return description



# Endpoint to handle user queries and return a response
@app.post("/query", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    # Check if a CSV has been uploaded
    if not is_csv_uploaded():
        print("CSV file is not uploaded.")
        return QueryResponse(response="Please upload a CSV file before asking a question.")

    # Define a retry limit to avoid infinite loops
    max_retries = 3
    attempts = 0
    
    while attempts < max_retries:
        try:
            # Attempt to locate and load the CSV file
            csv_file_path = os.path.join(UPLOAD_DIRECTORY, os.listdir(UPLOAD_DIRECTORY)[0])
            print(f"CSV file path: {csv_file_path}")

            df = pd.read_csv(csv_file_path)
            print(f"Dataframe loaded successfully. Number of rows: {len(df)}")

            # Check if the DataFrame is empty
            if df.empty:
                print("The uploaded CSV file is empty.")
                return QueryResponse(response="The uploaded CSV file is empty. Please upload a valid CSV file.")

            # Debugging the prompt received
            print(f"Received prompt: {request.prompt}")

            relevance_check = check_query_relevance(request.prompt, df)

            if relevance_check["related"] == "no":
                return QueryResponse(response=relevance_check["message"])

            # Generate Vega-Lite specification based on the prompt and dataframe
            vega_lite_spec = generate_chart(request.prompt, df)
            print(f"Generated Vega-Lite specification: {vega_lite_spec}")
            feedback = get_feedback(request.prompt, df, vega_lite_spec)

            final_spec = improve_response(request.prompt, df, vega_lite_spec, feedback)

            # Convert spec to JSON
            vega_spec = json.loads(final_spec)

            chart_description = generate_chart_description(request.prompt, final_spec)

            relevant_columns = vega_spec.get('data_columns', [])
            filters = vega_spec.get('filters', [])

            # Filter the DataFrame based on relevant columns and conditions
            if relevant_columns:
                df_filtered = df[relevant_columns]
            else:
                df_filtered = df

            # Apply any filters dynamically
            for condition in filters:
                df_filtered = df_filtered.query(condition)

            # Convert DataFrame to a list of dictionaries for Vega-Lite
            vega_spec['data'] = {'values': df_filtered.to_dict(orient='records')}
            # Set width for better visualization
            vega_spec['config'] = {
                'axisX': {
                    'labelOverlap': 'parity',  # Reduce overlap on the x-axis labels
                    'labelAngle': -45  # Optional: angle labels for better fit
                }
            }

            return QueryResponse(response=chart_description, vega_spec=vega_spec)

        except ValueError as value_error:
            print(f"ValueError during chart generation: {value_error}")
            attempts += 1
            print(f"Retrying chart generation (attempt {attempts}/{max_retries})...")
            if attempts >= max_retries:
                # If the maximum number of retries is reached, raise an HTTP exception
                raise HTTPException(status_code=500, detail="An error occurred during chart generation. Maximum retries reached.")
        except FileNotFoundError as fnf_error:
            print(f"FileNotFoundError: {fnf_error}")
            raise HTTPException(status_code=500, detail="CSV file not found. Please upload a file.")
        except pd.errors.EmptyDataError as empty_csv_error:
            print(f"Empty CSV file error: {empty_csv_error}")
            raise HTTPException(status_code=500, detail="CSV file is empty or corrupt.")
        except Exception as e:
            print(f"General exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))



# Endpoint for CSV upload
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")
    
    try:
        # Before saving the new file, delete any existing files in the 'uploads' directory
        if not os.path.exists(UPLOAD_DIRECTORY):
            os.makedirs(UPLOAD_DIRECTORY)

        # Clear previous files
        for existing_file in os.listdir(UPLOAD_DIRECTORY):
            file_path = os.path.join(UPLOAD_DIRECTORY, existing_file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {file_path}")

        # Save the new file
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        print(f"File saved at: {file_location}")
        return {"message": f"File '{file.filename}' uploaded successfully!"}
    
    except Exception as e:
        print(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload the file.")
