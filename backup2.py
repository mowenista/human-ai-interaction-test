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
import sys
from io import StringIO
import re

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

@app.post("/clear-chat")
async def clear_chat():

    return FileResponse('static/index.html')


@app.post("/clear_chat")
async def clear_chat():
    # Logic to clear any stored chat history if needed (e.g., clearing session or database)
    # For example: session.clear() if using session
    return {"message": "Chat history cleared"}

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


def get_generate_chart(query: str, df: pd.DataFrame) -> QueryResponse:
    print("generate chart")
    # Generate Vega-Lite specification based on the prompt and dataframe
    # Define a retry limit to avoid infinite loops
    max_retries = 3
    attempts = 0
    
    while attempts < max_retries:
        try:
            vega_lite_spec = generate_chart(query, df)
            print(f"Generated Vega-Lite specification: {vega_lite_spec}")
            feedback = get_feedback(query, df, vega_lite_spec)

            final_spec = improve_response(query, df, vega_lite_spec, feedback)

            # Convert spec to JSON
            vega_spec = json.loads(final_spec)

            chart_description = generate_chart_description(query, final_spec)

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


def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query
    
def get_data_analysis(query: str, df):
    print("Data Analysis")
    """
    Execute the given python code and return the output. 
    References:
    1. https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/utilities/python.py
    2. https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/tools/python/tool.py
    """
    # Generate a block of code to then answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f'''You are a helpful assistant that generates Python code to solve calculations or answer questions. 
                    Make sure that any result or answer is explicitly printed using print(...) so that the output 
                    can be captured and displayed. Do not return the result without print(...) because the code execution 
                    environment requires print(...) to capture output. Avoid extraneous explanations in the code, 
                    but ensure the code includes only what’s necessary to solve the problem and display results. The output should only be the python code, nothing else.'''
                ),
            },
            {
                "role": "user",
                "content": (
                    f'''Write Python code to solve the following: {query}. Use the dataframe, df, which has headers: {df.columns}
                    Make sure all results are printed using print(...) statements. Don't include any extra text, the output should only be python code. If the question asks for multiple answers respond in a tuple or dictionary'''
                ),
            },
        ],
        temperature=0.2,  # Lower temperature for more deterministic code generation
    )
    code = response.choices[0].message.content
    print(code)
    
    # Save the current standard output to restore later
    old_stdout = sys.stdout
    # Redirect standard output to a StringIO object to capture any output generated by the code execution
    sys.stdout = mystdout = StringIO()
    try:
        cleaned_command = sanitize_input(code)
        
        # Execute the sanitized command
        exec(cleaned_command)
        
        # Restore the original standard output after code execution
        sys.stdout = old_stdout
                
        # Capture only the last line of the output
        result = mystdout.getvalue().strip().splitlines()[-1]

        follow_up_prompt = (
            f"The result of the query '{query}' is: {result}. Can you give this response in a quick sentence. "
            
        )
        
        # Step 4: Get the LLM's response to the follow-up prompt
        follow_up_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an informative assistant that provides explanations and context based on data analysis results."
                    ),
                },
                {
                    "role": "user",
                    "content": follow_up_prompt,
                },
            ],
        )
        
        detailed_response = follow_up_response.choices[0].message.content
        
        # Step 5: Return both the result and the detailed response
        return QueryResponse(response=detailed_response)
            

            
    except Exception as e:
        sys.stdout = old_stdout
        print(f"General exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Tool section
generate_chart_tool = {
  "type": "function",
  "function": {
    "name": "get_generate_chart",
    "description": "Using the query from the user and the uploaded data, generate a chart using vegalite to visualize the question",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The users query on the included data set ",
            },
        },
        "additionalProperties": False,
    },
  }
}
data_analysis_tool = {
    "type": "function",
    "function": {
        "name": "get_data_analysis",
        "description": "Generate a block of code that would make the neccessary calculations that the users query requests",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A users query that could be answered by generating some python code",
                },
            },
            "additionalProperties": False,
        },
    }
}


tools = [generate_chart_tool, data_analysis_tool]
tool_map = {
    "generate_chart_tool": generate_chart_tool,
    "data_analysis_tool": data_analysis_tool
}
# Define the ReAct prompt based on your template
vanila_react_prompt = f'''
Answer the user question as best you can. You have access to the following tools:
generate_chart_tool: Using the query from the user and the uploaded data, generate a chart using vegalite to visualize the question
data_analysis_tool: Generate a block of code that would make the neccessary calculations that the users query requests

You run in a loop of Thought, Action, Observation in the following manner to answer the user question.

Question: the input question you must answer

Thought: you should always think about what to do.
Action: the tool name, should be one of [{tools}]. If no tool needed, just output "no tool".

You will return this action and action input, then wait for the Observation.

You will then call again with the result of the action.

Observation: the result of the action.
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Please ALWAYS start with a Thought.
'''

# React query processing function
def react_query(query, system_prompt, tool_map,df, max_iterations=3):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": query})
    i = 0

    while i < max_iterations:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        
        # Pattern to capture the action name and action input
        pattern = r'Action:\s*(.*?)\nAction Input:\s*(.*?)}'
        match = re.search(pattern, assistant_message, re.DOTALL)

        if match:
            action_name = match.group(1).strip()
            if action_name == "no tool":
                break
            
            
            # Call the appropriate function from the tool map
            result = tool_map[action_name](query, df)
            observation = f'Observation: action name: {action_name}, result: {result}'

            # Log the observation for debugging
            print(observation)

            # Append the observation to the messages
            messages.append({"role": "assistant", "content": f"Observation: {observation}"})
            continue
        else:
            break

    return assistant_message


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

            query = request.prompt
            out = react_query(query,vanila_react_prompt, tool_map, df,3)

            # Debugging the prompt received
            print(f"Received prompt: {query}")

            return QueryResponse(response=out)
            # prompt = f'''
            # The user provided a dataset with the following details (labels): {df.columns}
            # User Query: {query}
            # '''

            
            
            # tool_m = [
            #     {
            #         "role": "system", 
            #         "content": (
            #             "You are a data analytics assistant with access to two tools:\n"
            #             "1. **get_generate_chart** - Generate a Vega-Lite chart specification for visualization-related queries.\n"
            #             "2. **get_data_analysis** - Generate Python code to perform calculations or data analyses.\n\n"
            #             "Use the appropriate tool based on the user query context.\n"
            #             "- **If the user query is related to data visualization,** use `get_generate_chart`.\n"
            #             "- **If the query requires calculations or specific data manipulations,** use `get_data_analysis`.\n\n"
            #             "If the query does not seem relevant to either visualization or calculation, reply without using a tool."
            #         )
            #     },
            #     {"role": "user", "content": prompt}
            # ]

            # Replacing relevance check with tool call check
            # tool_check = client.chat.completions.create(
            #     model="gpt-4o-mini",
            #     messages=tool_m,
            #     tools=tools,
            # )

            # # Check if a tool call was made
            # if tool_check.choices[0].message.tool_calls==None:
            #     # No function call; return the response directly
            #     return QueryResponse(response=tool_check.choices[0].message.content)

            # func = tool_check.choices[0].message.tool_calls[0].function.name
            # # Check which function was called
            # if func == "get_generate_chart":
            #     return get_generate_chart(query, df)

            # elif func == "get_data_analysis":
            #     return get_data_analysis(query, df)

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
