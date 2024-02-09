import google.generativeai as genai
import os
import pandas as pd
import re
import logging
import streamlit as st

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Initialized Logger
logging.basicConfig(
    filename= "llm_plotting.log",
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: \
        %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)

#df = pd.read_excel('calls_4_jan.xlsx',sheet_name='PUTS')
#nlq = "Generates a bar chart of the strike price vs volume,open interest"

def generate_plot_code(df,nlq):
    try:
        logging.info(f"df attributes : {df.columns.to_list()}")
        model = genai.GenerativeModel('gemini-pro')

        prompt = f'''Write a Python function to generate a chart based on the following natural language query (NLQ):

        {nlq}

        DataFrame attributes: {df.columns.to_list()}.
        Constraints:
        1. Always use "df" as a parameter in your function.
        2. Utilize the Plotly library to plot charts whenever applicable.
        3. Ensure that the attributes mentioned in the NLQ are present in the DataFrame attributes list.
        4. Always import necessary libraries.
        5. Always call function at the end and store result of the function into a variable named "plot"

        Example:
        NLQ : Generate a line chart with combintation of Strike, Volume and Open Interest.
        Response:
        ```python
        import plotly.graph_objects as go

        def generate_chart(df):
            """
            Generates a chart of the strike price, volume, and open interest for a given stock.

            Args:
                df (pandas.DataFrame): The dataframe containing the stock data.

            Returns:
                plotly.graph_objects.Figure: The chart of the strike price, volume, and open interest.
            """

            if "STRIKE" not in df.columns or "VOLUME" not in df.columns or "OI" not in df.columns:
                raise ValueError("One or more of the specified columns do not exist in the dataframe.")

            # Create the figure
            fig = go.Figure()

            # Add the strike price trace
            fig.add_trace(go.Scatter(x=df["STRIKE"], y=df["STRIKE"], name="Strike Price"))

            # Add the volume trace
            fig.add_trace(go.Scatter(x=df["STRIKE"], y=df["VOLUME"], name="Volume"))

            # Add the open interest trace
            fig.add_trace(go.Scatter(x=df["STRIKE"], y=df["OI"], name="Open Interest"))

            # Set the layout
            fig.update_layout(title="Strike Price, Volume, and Open Interest", xaxis_title="Strike Price", yaxis_title="Value")

            return fig
        plot = generate_chart(df)
        ```
        '''
        logging.info(f'\n{prompt}\n')
        response = model.generate_content(prompt)
        #logging.info(f'\n{response.text}\n')

        pattern = r'```python(.*?)```'

        # Extract SQL code using regular expression
        matches = re.findall(pattern, response.text, re.DOTALL)

        if matches:
            code = matches[0].strip()
            logging.info(f'\n{code}')
        else:
            logging.exception("\n ****** No Python code found from an LLM. ******")

        return code
    except Exception as e:
        logging.exception(f'\n {str(e)}')

# Function to read CSV/Excel file
@st.cache_data
def read_file(file):
    if str(file.name).endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    return df

try:
    st.set_page_config(page_title='Plotify-NLQ')
    # App layout
    st.title("NLQ Graph Generator")
    logging.info("\n*********************** LOGGING STARTED **********************************\n")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Read the uploaded file
        df = read_file(uploaded_file)
        st.write("Uploaded file preview:")
        st.write(df.head())

        # Text box for NLQ
        nlq = st.text_area("Enter your natural language query (NLQ):")
        # Button to generate chart
        if st.button("Generate Chart"):
            if nlq:
                with st.spinner(text="Generating plot..."):
                    code = generate_plot_code(df,nlq)
                    try:
                        exec(code)
                    except Exception as e:
                        logging.exception(f'{str(e)}')
                    fig = plot
                    #result.write_html('sample_plot.html')
                    #st.components.v1.html(result.to_html(),height=800, scrolling=True)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please enter a NLQ.")

except Exception as e:
    st.exception(e)
    logging.exception(f'\n {str(e)}')
