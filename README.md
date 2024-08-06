# Union_Budget_2024_InforBot

This project is an RAG application that analyzes the Indian Union Budget 2024-2025.
It leverages Context based and Dataframe based retrieval technique to provide intelligent insights, budget data accessible and understandable.


## Key Attributes

- **LLM-Powered Analysis**: Utilizes LangChain and Large Language Models to provide intelligent insights on budget data.
- **Interactive Web Interface**: User-friendly Streamlit app for easy access and interaction.
- **Concurrent Query Handling**: Implements ThreadPoolExecutor for efficient processing of multiple queries.
- **Present Thought of Action**: Displays Callbacks for understanding the Thought process and Observation of the Bot
- **Chat History**: Utilize chat history for in depth analysis.
- **OpenAI/Ollama**: Can work with Open AI API or offline based models from Ollama such as Gemma2 and Llama3

## Modules Utilized

- **LangChain**: For building the AI agent, tools and query processing.
- **Streamlit**: Creates the web interface.
- **OpenAI GPT / Llama 3 (via Ollama)**: Large Language model for analysis.
- **Python**: Core programming language.
- **Matplotlib/Seaborn**: For data visualization.
- **Pandas**: For data manipulation and analysis.

> #### To Do
> - Implement Tax Calculator
> - Implement Dashboard with visualizations
> - Implement Visualization Tools to the agent
> - Better UX/UI and Open AI Key Insert at Web client
> - Improve Model Prompt and Document Prompts


### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
2. Install the required packages:

`pip install -r requirements.txt`

3. Set up your OpenAI API key 
    1. Create a .env file in your working directory
    2. Write your OpenAI API key
    
    `OPENAI_API_KEY = '<your open ai API key>'`

4. Ollama installation (for Llama 3) - [Ollama Download](https://ollama.com/download).

## Screenshots

![Thinking](https://github.com/VigneshArjunRaj/Union_Budget_2024_InforBot/blob/main/Outputs/Output1.png)

![Complete](https://github.com/VigneshArjunRaj/Union_Budget_2024_InforBot/blob/main/Outputs/Output3.png)

![Callback](https://github.com/VigneshArjunRaj/Union_Budget_2024_InforBot/blob/main/Outputs/Output2.png)

![Terminal](https://github.com/VigneshArjunRaj/Union_Budget_2024_InforBot/blob/main/Outputs/Output1.png)

## Data Sources and References

The application uses Research data and few official documents from the Indian Union Budget 2024-2025, including:
- Central Government Revenue Analysis Tool - [PRS Legislative Research](https://prsindia.org/budgets/parliament/interim-union-budget-2024-25-analysis)
- Textual Previous Year Budget Analysis - [PRS Legislative Research Union Budget 23-24](https://prsindia.org/files/budget/budget_parliament/2023/Union_Budget_Analysis-2023-24.pdf)
- Textual Budget Analysis - [PRS Legislative Research](https://prsindia.org/budgets/parliament/interim-union-budget-2024-25-analysis)
- Budget Speech Document from Finance Minister[Budget Speech](https://www.indiabudget.gov.in/doc/budget_speech.pdf)
- Subsidaries - [PRS Legislative Research](https://prsindia.org/budgets/parliament/interim-union-budget-2024-25-analysis)
- Central Government Expenditure Breakdown - [PRS Legislative Research](https://prsindia.org/budgets/parliament/interim-union-budget-2024-25-analysis)
- Social Welfare Expenditure - [PRS Legislative Research](https://prsindia.org/budgets/parliament/interim-union-budget-2024-25-analysis)
- Allocation for Schemes - [PRS Legislative Research](https://prsindia.org/budgets/parliament/interim-union-budget-2024-25-analysis)
- Ministry-wise expenditure - [PRS Legislative Research](https://prsindia.org/budgets/parliament/interim-union-budget-2024-25-analysis)
- Scheme-wise allocations - [PRS Legislative Research](https://prsindia.org/budgets/parliament/interim-union-budget-2024-25-analysis)
- Tax revenue breakdowns - [PRS Legislative Research](https://prsindia.org/budgets/parliament/interim-union-budget-2024-25-analysis)
- Deficit targets - [PRS Legislative Research](https://prsindia.org/budgets/parliament/interim-union-budget-2024-25-analysis)
- Financials of Central Government - [PRS Legislative Research Discussion Paper](https://prsindia.org/files/budget/Finance_of_the_Central_Government_2019-20_to_2024-25.pdf)



## Usage

1. Enter your budget-related question in the text input field.
2. The AI will process your query and provide an analysis.

### Running the Application

1. Start the Streamlit app:
streamlit run Application.py

## Disclaimer

This is an experimental project designed to explore the capabilities of Large Language Models (LLMs) in interpreting and analyzing complex financial data.
It is important to note that this tool is not created, endorsed, or verified by any official financial institution, government body, or professional financial advisory group.
The AI model used in this application may occasionally generate inaccurate or nonsensical responses, especially when dealing with nuanced financial concepts or specific budget details.
The primary goal of this project is to test and demonstrate the analytical potential of LLMs when applied to structured financial data, rather than to provide authoritative budget analysis or financial advice.
Users should approach the tool's outputs with critical thinking and verify any important information with official sources.
This tool should not be used as a basis for financial decisions or official budget interpretations.

