from PromptandRetriever import Prompt
from langchain_ollama import ChatOllama
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from AgentTools import AgentTools



def init_agent_executor(model_name = "OpenAI"):
    """_summary_

    Args:
        model_name (str, optional): Model name for Agent LLM. Defaults to "OpenAI".
    
    Initialize LLM Models and agent tools, binding all of them together with Output Parser and Prompt Template.

    Returns:
        _type_: AgentExecutor
    """    

    # Choose Between Ollama Offline models such as Gemma2/Llamma3 or OpenAI's GPT4o-mini
    if model_name == "Gemma2":
        openai4o_llm = ChatOllama(model="gemma2:2b")
    elif model_name == "llama3":
        openai4o_llm = ChatOllama(model="llama3:8b")
    else:
        openai4o_llm = ChatOpenAI(model="gpt-4o-mini")
    

    # Initialize AgentTools class for creating tools
    AgentFactor = AgentTools()

    # Pandas/CSV tools for Data based RAG
    SchemeAllocationAgen_tool = AgentFactor.SchemeAllocationAgen(openai4o_llm)
    BudgetGlanceAgent_tool = AgentFactor.BudgetGlanceAgent(openai4o_llm)
    CentralBreakupAgent_tool = AgentFactor.CentralBreakupAgent(openai4o_llm)
    CentralExpensesAgent_tool = AgentFactor.CentralExpensesAgent(openai4o_llm)
    FRBMAgent_tool = AgentFactor.FRBMAgent(openai4o_llm)
    MinistryAgent_tool = AgentFactor.MinistryAgent(openai4o_llm)
    TaxSlabsAgent_tool = AgentFactor.TaxSlabsAgent(openai4o_llm)
    SpecialAllocationAgent_tool = AgentFactor.SpecialAllocationAgent(openai4o_llm)

    # Text Embeddings tools for Context based RAG
    BudgetSpeech = AgentFactor.BudgetSpeech()
    BudgetAnalysis = AgentFactor.BudgetAnalysis()
    BudgetAnalysisPrevious = AgentFactor.BudgetAnalysisPrevious()
    Financials =AgentFactor.FinancialsOfCentralGovernment()
    ExpertsOpinion = AgentFactor.ExpertsOpinion()

    # Collecting all tools in agent_as_tools
    agent_as_tools = [
    BudgetSpeech,
    BudgetAnalysis,
    BudgetAnalysisPrevious,
    BudgetGlanceAgent_tool,
    CentralBreakupAgent_tool,
    CentralExpensesAgent_tool,
    SchemeAllocationAgen_tool,
    FRBMAgent_tool,
    MinistryAgent_tool,
    TaxSlabsAgent_tool,
    SpecialAllocationAgent_tool,
    Financials,
    ExpertsOpinion,
    ]


    # Bind the LLM model with Observation stopping
    chat_model_with_stop = openai4o_llm.bind(stop=["```\n\nObservation:"])

    # Add Tools and properties of tools to the prompt
    prompt = Prompt.partial(
        tools=agent_as_tools,
        tool_names=", ".join([t.name for t in agent_as_tools]),
        tools_description = " ".join([t.name + ": \n" + t.description +"\n \n " for t in agent_as_tools])
    )
    
    # Agent Chaining with Input - Mpping, Prompt, Observation Stopping Model and ReAct OutputParser
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"]

        }
        | prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
    )

    # instantiate AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=agent_as_tools, verbose=True,handle_parsing_errors=True)

    return agent_executor



