### NLQ to Chart Generation using LLM

This repository contains code for a Streamlit web application that allows users to generate charts from natural language queries (NLQ) using a Language Model (LLM). The application provides an intuitive interface where users can upload their data, input NLQ queries, and instantly visualize their insights in the form of interactive charts.

#### Features:
- **NLQ Interface**: Users can input NLQ queries to generate various types of charts, including bar charts, line charts, scatter plots, etc.
- **Data Upload**: Supports uploading CSV or Excel files containing the data to be visualized.
- **Interactive Visualization**: Generated charts are interactive, allowing users to explore and analyze the data easily.
- **Efficient Chart Generation**: Utilizes a pre-trained Language Model (LLM) to interpret NLQ queries and generate corresponding charts quickly and accurately.
- **Streamlit Integration**: The application is built using Streamlit, making it easy to deploy and share with others.

#### Usage:
1. Upload your data file (CSV or Excel).
2. Enter your natural language query (NLQ) to specify the desired chart.
3. Click on the "Generate Chart" button to visualize your data instantly.

#### Getting Started:
To run the application locally:
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run plotting_app.py`
