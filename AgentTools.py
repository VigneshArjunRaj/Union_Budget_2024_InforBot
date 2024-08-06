from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain.tools import StructuredTool
from EmbeddingModel import EmbeddingsDataModel
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import PromptTemplate

class AgentTools:
    """_summary_

    Contains Methods and Attributes to create Pandas and Data/Context based Agents

    Contains Name, Descriptions and Document Prompts for Each Agent
    """
    def __init__(self):
        self.embeddings = EmbeddingsDataModel.getEmbeddings()
        self.dataclass = EmbeddingsDataModel()
        print("Please Initialize the Agents")
    
    # Pandas Dataframe based tools creator
    def createAgent(self,data,llm,name,description):
        """_summary_

        Args:
            data (Dataframe): Dataframe
            llm (ChatOpenAI | OllamaChat | GroqChat): LLM Model
            name (str): Name of the Tool
            description (str): Description of the Tool

        Returns:
            StructuredTool: Tool for Dataframe
        """
        # Create Pandas Based Agent
        agent = create_pandas_dataframe_agent(llm,data,'tool-calling',allow_dangerous_code=True)
        # COnvert the Agent to Structured Tool
        agent_as_tool = StructuredTool.from_function(agent.run, name=name,description=description)
        return agent_as_tool
    
    # Context/ Data Based tools Creator
    def createTextAgent(self,vectorstore,name,description):
        """_summary_

        Args:
            vectorstore (FAISS): _description_
            name (str): Name of the Tool
            description (str): Description of the Tool

        Returns:
            Tool: tool for Context
        """        
        retriever = vectorstore.as_retriever()
        retriever_tool = create_retriever_tool(retriever,name = name,description=description, document_prompt= PromptTemplate(input_variables=["context"],template = """
                Answer based on the following context
                {context}
                """))
        return retriever_tool
    
    def SubsidiesAgent(self,llm):
        """_summary_

        Args:
            llm (ChatOpenAI | OllamaChat | GroqChat): LLM Model

        Subsidy Analysis Tool

        Returns:
            Structured Tool
        """    
        data = self.dataclass.getSubsidies()
        agent_as_tool = self.createAgent(data,llm, name = "Subsidy Analysis Tool",
                                description= """This tool provides detailed information and analysis on subsidies in the Indian Union Budget for the fiscal year 2024-25.
                                It contains data on five major subsidy categories: Food subsidy, Fertiliser subsidy, Interest subsidy, LPG subsidy, and Other subsidies.
                                The tool offers comparative data across multiple fiscal years and budget estimates:

                                Actuals for 2022-23
                                Budgeted estimates for 2023-24
                                Revised estimates for 2023-24
                                Budgeted estimates for 2024-25
                                Additionally, it calculates the percentage change between the revised estimates of 2023-24 and the budgeted estimates of 2024-25.
                                Capabilities:

                                Retrieve specific subsidy amounts for any category and fiscal year/estimate.
                                Calculate year-over-year changes in subsidy allocations.
                                Provide insights on budget priorities based on subsidy allocations.
                                Compare different types of subsidies within the same fiscal year.
                                Analyze trends in subsidy allocations over the given time period.  """)
        return agent_as_tool
    
    def SpecialAllocationAgent(self,llm):
        """_summary_

        Args:
            llm (ChatOpenAI | OllamaChat | GroqChat): LLM Model

        Social Welfare Expenditure Analysis Tool

        Returns:
            Structured Tool
        """  
        data = self.dataclass.getSpecialAllocationData()
        agent_as_tool = self.createAgent(data,llm, name = "Social Welfare Expenditure Analysis Tool",
                                            description= """This tool provides comprehensive information and analysis on government expenditure for various social welfare schemes in the Indian Union Budget. 
                                            It focuses on four key areas of social welfare: Scheduled Caste sub-plans, Scheduled Tribe sub-plans, Schemes for the welfare of women and children, Schemes for the North Eastern Region

                                            The tool offers comparative data across multiple fiscal years and budget estimates:

                                            Actuals for 2022-23
                                            Revised estimates for 2023-24
                                            Budgeted estimates for 2024-25

                                            Additionally, it calculates the percentage change between the actuals of 2022-23 and the budgeted estimates of 2024-25, providing insight into the government's changing priorities and commitments to these social welfare areas.
                                            Capabilities:

                                            Retrieve specific expenditure amounts for any category and fiscal year/estimate.
                                            Calculate changes in allocations between actual spending and future budgets.
                                            Provide insights on the government's priorities in social welfare based on allocation trends.
                                            Compare expenditures across different social welfare categories within the same fiscal year.
                                            Analyze trends in social welfare spending over the given time period.
                                            Assess the government's commitment to marginalized communities and specific regions.

 """)
        return agent_as_tool
        



        
    def CentralExpensesAgent(self,llm):
        """_summary_

        Args:
            llm (ChatOpenAI | OllamaChat | GroqChat): LLM Model

        Central Government Expenditure Breakdown Analysis Tool

        Returns:
            Structured Tool
        """  
        data = self.dataclass.getCentralExpensesData()
        agent_as_tool = self.createAgent(data,llm,name = "Central Government Expenditure Breakdown Analysis Tool",
                                        description= """This tool provides a detailed breakdown and analysis of the central government's expenditure in India for the fiscal year 2024-25.
                                          It covers various categories of government spending, including:
                                            Central Expenditure, Establishment Expenditure of Centre, Central Sector Schemes,
                                            Other expenditure (with a focus on interest payments), Grants for Centrally Sponsored Schemes (CSS) and other transfers,
                                            Centrally Sponsored Schemes (CSS), Finance Commission Grants, further broken down into:
                                            Rural Local Bodies, Urban Local Bodies, Disaster Management Grants,
                                            Post Devolution Revenue Deficit Grants, Other grants, loans and transfers,

                                            The tool offers comparative data across multiple fiscal years and budget estimates:

                                            Actuals for 2022-23,
                                            Revised estimates for 2023-24,
                                            Budgeted estimates for 2024-25,

                                            It also calculates the percentage change between the actuals of 2022-23 and the budgeted estimates of 2024-25, providing insights into spending trends and shifts in budget allocations.
                                            Capabilities:

                                            Retrieve specific expenditure amounts for any category and fiscal year/estimate.
                                            Calculate changes in allocations between actual spending and future budgets.
                                            Provide insights on the government's spending priorities across different sectors and schemes.
                                            Compare expenditures across various categories within the same fiscal year.
                                            Analyze trends in government spending over the given time period.
                                            Assess the distribution of funds between central schemes and transfers to local bodies.
                                            Evaluate the focus on different aspects of governance, from establishment costs to development schemes.""")
        return agent_as_tool
    

    def CentralBreakupAgent(self,llm):
        """_summary_

        Args:
            llm (ChatOpenAI | OllamaChat | GroqChat): LLM Model

        Central Government Revenue Analysis Tool

        Returns:
            Structured Tool
        """  
        
        data = self.dataclass.getCentralBreakupData()
        agent_as_tool = self.createAgent(data,llm,name = "Central Government Revenue Analysis Tool",
                                                     description= """This tool provides a comprehensive breakdown and analysis of the central government's receipts in India for the fiscal year 2024-25.
                                                       It covers various categories of government revenue, including:
                                    A. Gross Tax Revenue

                                    Corporation Tax
                                    Taxes on Income
                                    Goods and Services Tax
                                    Customs
                                    Union Excise Duties
                                    Service Tax
                                    B. Devolution to States
                                    C. Centre's Net Tax Revenue
                                    D. Non-Tax Revenue
                                    Interest Receipts
                                    Dividend
                                    Other Non-Tax Revenue
                                    E. Capital Receipts (without borrowings)
                                    Disinvestment
                                    F. Total Receipts (without borrowings) (C+D+E)

                                    The tool offers comparative data across multiple fiscal years and budget estimates:

                                    Actuals for 2022-23
                                    Revised estimates for 2023-24
                                    Budgeted estimates for 2024-25

                                    It also calculates the percentage change between the actuals of 2022-23 and the budgeted estimates of 2024-25, providing insights into revenue trends and shifts in the government's financial inflows.
                                    Capabilities:

                                    """)
        return agent_as_tool
    

    def BudgetGlanceAgent(self,llm):
        """_summary_

        Args:
            llm (ChatOpenAI | OllamaChat | GroqChat): LLM Model

        BudgetGlance

        Returns:
            Structured Tool
        """ 

        data = self.dataclass.getBudgetGlanceData()
        agent_as_tool = self.createAgent(data,llm,name = "BudgetGlance",
                        description= """
This tool provides a glance of total Budget  in India for the fiscal year 2024-25. It covers the following 
Revenue Expenditure
Capital Expenditure
Capital Outlay
Loans and Advances
Total Expenditure
Revenue Receipts
Capital Receipts
Recoveries of Loans
Other receipts (including disinvestments)
Total Receipts (excluding borrowings)
Revenue Deficit
Revenue Deficit % of GDP
Fiscal Deficit
Fiscal Deficit % of GDP
Primary Deficit
Primary Deficit % of GDP

The tool offers comparative data across multiple fiscal years and budget estimates:

                            Actuals for 2022-23
                            Revised estimates for 2023-24
                            Budgeted estimates for 2024-25

                            Additionally, it calculates the percentage change between the actuals of 2022-23 and the budgeted estimates of 2024-25, providing insight into the government's fiscal consolidation efforts and adherence to FRBM targets.
                            
""")
        return agent_as_tool
    

    def FRBMAgent(self,llm):
        """_summary_

        Args:
            llm (ChatOpenAI | OllamaChat | GroqChat): LLM Model

        FRBM Target Analysis Tool

        Returns:
            Structured Tool
        """ 
        data = self.dataclass.getFRBMData()
        agent_as_tool = self.createAgent(data,llm,name = "FRBM Target Analysis Tool",
                                                     description= """
                            This tool provides analysis and tracking of India's Fiscal Responsibility and Budget Management (FRBM) targets for key deficit indicators as a percentage of GDP. It focuses on three critical deficit metrics:

                            Fiscal Deficit
                            Revenue Deficit
                            Primary Deficit

                            The tool offers comparative data across multiple fiscal years and budget estimates:

                            Actuals for 2022-23
                            Revised estimates for 2023-24
                            Budgeted estimates for 2024-25

                            Additionally, it calculates the percentage change between the actuals of 2022-23 and the budgeted estimates of 2024-25, providing insight into the government's fiscal consolidation efforts and adherence to FRBM targets.
                            

                            """)
        return agent_as_tool
    

    def MinistryAgent(self,llm):
        """_summary_

        Args:
            llm (ChatOpenAI | OllamaChat | GroqChat): LLM Model

        Ministry-wise Expenditure Analysis Tool

        Returns:
            Structured Tool
        """ 
        data = self.dataclass.getMinistryData()
        agent_as_tool = self.createAgent(data,llm,name = "Ministry-wise Expenditure Analysis Tool",
                                         description= """This tool provides a detailed breakdown and analysis of the Indian government's expenditure across various ministries for the fiscal year 2024-25. It covers the following ministries and categories:

                                        Defence
                                        Road Transport and Highways
                                        Railways
                                        Consumer Affairs, Food and Public Distribution
                                        Home Affairs
                                        Rural Development
                                        Chemicals and Fertilisers
                                        Communications
                                        Agriculture and Farmers' Welfare
                                        Education
                                        Jal Shakti
                                        Health and Family Welfare
                                        Housing and Urban Affairs
                                        Other Ministries
                                        Total Expenditure

                                        The tool offers comparative data across multiple fiscal years and budget estimates:

                                        Actuals for 2022-23
                                        Revised estimates for 2023-24
                                        Budgeted estimates for 2024-25

                                        It also calculates the percentage change between the actuals of 2022-23 and the budgeted estimates of 2024-25, providing insights into spending trends and shifts in budget allocations across ministries.
                                        Capabilities:

                                        """)
        return agent_as_tool
    

    def SchemeAllocationAgen(self,llm):
        """_summary_

        Args:
            llm (ChatOpenAI | OllamaChat | GroqChat): LLM Model

        Scheme-wise Allocation Analysis Tool

        Returns:
            Structured Tool
        """ 
        data = self.dataclass.getSchemeAllocationData()
        agent_as_tool = self.createAgent(data,llm,name = "Scheme-wise Allocation Analysis Tool",
                                                     description= """This tool provides a detailed breakdown and analysis of the Indian government's expenditure on various major schemes for the fiscal year 2024-25. It covers the following schemes:

                                        MGNREGS (Mahatma Gandhi National Rural Employment Guarantee Scheme)
                                        Pradhan Mantri Awas Yojana
                                        Jal Jeevan Mission/National Rural Drinking Water Mission
                                        PM-KISAN
                                        National Health Mission
                                        National Education Mission
                                        Modified Interest Subvention Scheme
                                        Saksham Anganwadi and POSHAN 2.0
                                        National Livelihood Mission-Ajeevika
                                        Pradhan Mantri Fasal Bima Yojana
                                        Reform Linked Distribution Scheme
                                        PM POSHAN
                                        Swachh Bharat Mission
                                        Pradhan Mantri Gram Sadak Yojana
                                        Pradhan Mantri Krishi Sinchai Yojana

                                        The tool offers comparative data across multiple fiscal years and budget estimates:

                                        Actuals for 2022-23
                                        Revised estimates for 2023-24
                                        Budgeted estimates for 2024-25

                                        It also calculates the percentage change between the actuals of 2022-23 and the budgeted estimates of 2024-25, providing insights into spending trends and shifts in budget allocations across these key government schemes.
                                        """)
        return agent_as_tool
    

    def TaxSlabsAgent(self,llm):
        """_summary_

        Args:
            llm (ChatOpenAI | OllamaChat | GroqChat): LLM Model

        Income Tax Slab Analysis and Calculation Tool

        Returns:
            Structured Tool
        """

        data = self.dataclass.getSpecialAllocationData()
        agent_as_tool = self.createAgent(data,llm,name = " Income Tax Slab Analysis and Calculation Tool",
                                                     description= """This tool provides information about the tax slabs in India, comparing the current income tax structure with the proposed changes.
                                                       It contains data on tax rates and corresponding income slabs for both the current and proposed systems.
                                                         The tool can be used to analyze changes in the tax structure and calculate tax liabilities based on given income levels.
                                            The tool includes the following data:

                                            Tax Rate: The percentage of tax applicable for each slab
                                            Current Income Slab: The income ranges for the existing tax structure
                                            Proposed Income Slab: The income ranges for the proposed tax structure

                                            
                                            """)
        return agent_as_tool
    

    # Text Based Tools
    def BudgetSpeech(self):
        """_summary_

        

        Budget Speech Document from Finance Minister

        Returns:
            Tool
        """
        vectorstore = self.dataclass.getBudgetSpeech()
        agent_as_tool = self.createTextAgent(
            vectorstore,name = "Budget Speech Document from Finance Minister",
            description= """This tool contains the full text of the Union Budget speech delivered by the Finance Minister.
            It provides a narrative overview of the government's fiscal policies, priorities, and major announcements for the 2024-2025 fiscal year.
            """)
        return agent_as_tool
    

    def ExpertsOpinion(self):
        """_summary_

        

        Opinions and Analyses from financial experts and market leaders
        Returns:
            Tool
        """
        vectorstore = self.dataclass.getExpertsOpinion()
        agent_as_tool = self.createTextAgent(vectorstore,name = "Opinions and Analyses from financial experts and market leaders ",
                                                     description= """This tool compiles opinions and analyses from financial experts and market leaders regarding the 2024-2025 Union Budget.
                                                       It offers diverse perspectives on the potential impacts and implications of the budget.

                                Capabilities:

                                Provide expert interpretations of budget measures
                                Offer insights into potential market reactions
                                Highlight areas of consensus or disagreement among experts""")
        return agent_as_tool
    

    def BudgetAnalysis(self):
        """_summary_

        

        Textual Analysis of the current Budget

        Returns:
            Structured Tool
        """
        vectorstore = self.dataclass.getBudgetAnalysis()
        agent_as_tool = self.createTextAgent(vectorstore,name = "Textual Analysis of the current Budget",
                                                     description= """This tool contains comprehensive textual analysis of the current budget.
                             It provides in-depth explanations and interpretations of budget allocations, changes, and their potential impacts.
                             If Other tools doesn't help, this tool can be used regarding the Budget for 2024

                               """)
        return agent_as_tool
    

    def BudgetAnalysisPrevious(self):
        """_summary_

        

        Textual Analysis of the Previous Budget

        Returns:
            Structured Tool
        """
        vectorstore = self.dataclass.getBudgetAnalysisPrevious()
        agent_as_tool = self.createTextAgent(vectorstore,name = "Textual Analysis of the Previous Budget",
                                                     description= """This tool contains comprehensive textual analysis of the previous year budget (2023).
                             It provides in-depth explanations and interpretations of budget allocations, changes, and their potential impacts.
                             If Other tools doesn't help, this tool can be used regarding the Budget for 2023

                                   """)
        return agent_as_tool
    
    def FinancialsOfCentralGovernment(self):
        """_summary_

        

        Textual Analysis of India's Central Government Finances

        Returns:
            Structured Tool
        """
        vectorstore = self.dataclass.getFinancialsOfCentralGovernment()
        agent_as_tool = self.createTextAgent(vectorstore,name = "Textual Analysis of India's Central Government Finances",
                                                     description= """This tool provides a textual analysis of India's central government finances from 2019 to 2024. 
                                                     It offers a broader perspective on the country's fiscal trends over a five-year period. """)
        return agent_as_tool
    


