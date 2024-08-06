import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
"""
Model File for Accessing, Storing and Retrieving Data Models
"""

#static files path for Data

SpecialAllocationDataPath = "static_files/Allocation for specials 24-25.csv"
CentralExpensesDataPath = "static_files/Break up Exp central 24-25.csv"
CentralBreakupDataPath = "static_files/Break up of central 24-25.csv"
BudgetGlanceDataPath = "static_files/Budget at a Glance 2024 - 25.csv"
FRBMDataPath = "static_files/FRBM.csv"
MinistryDataPath = "static_files/Ministry-wise exp 24-25.csv"
SchemeAllocationDataPath = "static_files/scheme wise allocation 23-24.csv"
TaxSlabsDataPath = "static_files/Tax Slabs.csv"
ExpertsOpinionPath = "static_files/ExpertReactions.txt"
BudgetAnalysisPath = "static_files/Union_Budget_analysis_24-25.txt"
BudgetAnalysisPreviousPath = "static_files/Union_budget_analysis_23-24.txt"
BudgetSpeechPath = "static_files/budgetspeech.txt"
FinancialsOfCentralGovernmentPath = "static_files/Financialsofcentralgovernment2019to25.txt"
SubsidiesDataPath = "static_files/Subsidies in 2024-25.csv"

class EmbeddingsDataModel:
    """_summary_
    Contains Methods and Attributes to Access and Modify data for Tool Creation and Visualization
    """
    def __init__(self):
        print("Embeddings are being Initialized")
    
    def convertTextToDocuments(self,text_path: str):
        """_summary_

        Args:
            text_path (str | Path): _description_
        

        Returns:
            List[Document] : Text from file to List of Documents
        """
        documents_api = TextLoader(text_path,encoding = 'UTF-8').load()

        # Recursively splitting documents into chunks with overlaps
        recursivesplit = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        documents_post_split = recursivesplit.split_documents(documents_api)
        return documents_post_split
    
    def getExpertsOpinion(self):
        """_summary_
        Returns FAISS Embeddings with Experts Opions context
        Returns:
            FAISS: Embeddings
        """
        
        if os.path.exists("LocalEmbeddings/expertsOpinion"):
            vectorstore = FAISS.load_local("LocalEmbeddings/expertsOpinion", EmbeddingsDataModel.getEmbeddings(),allow_dangerous_deserialization=True)
            if not vectorstore:
                docs = self.convertTextToDocuments(ExpertsOpinionPath)

                vectorstore = FAISS.from_documents(docs, EmbeddingsDataModel.getEmbeddings())
                #store vector db 
                vectorstore.save_local(folder_path="LocalEmbeddings/expertsOpinion")
        else:
            # returns vector store db from documents and embeddings
            docs = self.convertTextToDocuments(ExpertsOpinionPath)
            vectorstore = FAISS.from_documents(docs, EmbeddingsDataModel.getEmbeddings())
            #store vector db 
            vectorstore.save_local(folder_path="LocalEmbeddings/expertsOpinion")
        # return The FAISS Embeddings Vector STore DB
        return vectorstore

    def getBudgetAnalysisPrevious(self):
        """_summary_
        Returns FAISS Embeddings with Previous Fiscal Year Budget Analysis context
        Returns:
            FAISS: Embeddings
        """
        
        if os.path.exists("LocalEmbeddings/BudgetAnalysisPrevious"):
            vectorstore = FAISS.load_local("LocalEmbeddings/BudgetAnalysisPrevious",  EmbeddingsDataModel.getEmbeddings(),allow_dangerous_deserialization=True)
            if not vectorstore:
                docs = self.convertTextToDocuments(BudgetAnalysisPreviousPath)

                vectorstore = FAISS.from_documents(docs, EmbeddingsDataModel.getEmbeddings())
                #store vector db 
                vectorstore.save_local(folder_path="LocalEmbeddings/BudgetAnalysisPrevious")
        else:
            # returns vector store db from documents and embeddings
            docs = self.convertTextToDocuments(BudgetAnalysisPreviousPath)
            vectorstore = FAISS.from_documents(docs, EmbeddingsDataModel.getEmbeddings())
            #store vector db 
            vectorstore.save_local(folder_path="LocalEmbeddings/BudgetAnalysisPrevious")
        # return The FAISS Embeddings Vector STore DB
        return vectorstore
    
    def getBudgetSpeech(self):
        """_summary_
        Returns FAISS Embeddings with Budget Speech context
        Returns:
            FAISS: Embeddings
        """
        
        if os.path.exists("LocalEmbeddings/BudgetSpeech"):
            vectorstore = FAISS.load_local("LocalEmbeddings/BudgetSpeech",  EmbeddingsDataModel.getEmbeddings(),allow_dangerous_deserialization=True)
            if not vectorstore:
                docs = self.convertTextToDocuments(BudgetSpeechPath)

                vectorstore = FAISS.from_documents(docs, EmbeddingsDataModel.getEmbeddings())
                #store vector db 
                vectorstore.save_local(folder_path="LocalEmbeddings/BudgetSpeech")
        else:
            docs = self.convertTextToDocuments(BudgetSpeechPath)
            # returns vector store db from documents and embeddings
            vectorstore = FAISS.from_documents(docs, EmbeddingsDataModel.getEmbeddings())
            #store vector db 
            vectorstore.save_local(folder_path="LocalEmbeddings/BudgetSpeech")
        # return The FAISS Embeddings Vector STore DB
        return vectorstore

    def getBudgetAnalysis(self):
        """_summary_
        Returns FAISS Embeddings with Budget Analysis context
        Returns:
            FAISS: Embeddings
        """
        
        if os.path.exists("LocalEmbeddings/BudgetAnalysis"):
            vectorstore = FAISS.load_local("LocalEmbeddings/BudgetAnalysis",  EmbeddingsDataModel.getEmbeddings(),allow_dangerous_deserialization=True)
            if not vectorstore:
                docs = self.convertTextToDocuments(BudgetAnalysisPath)

                vectorstore = FAISS.from_documents(docs, EmbeddingsDataModel.getEmbeddings())
                #store vector db 
                vectorstore.save_local(folder_path="LocalEmbeddings/BudgetAnalysis")
        else:
            docs = self.convertTextToDocuments(BudgetAnalysisPath)
            # returns vector store db from documents and embeddings
            vectorstore = FAISS.from_documents(docs, EmbeddingsDataModel.getEmbeddings())
            #store vector db 
            vectorstore.save_local(folder_path="LocalEmbeddings/BudgetAnalysis")
        # return The FAISS Embeddings Vector STore DB
        return vectorstore
    
    def getFinancialsOfCentralGovernment(self):
        """_summary_
        Returns FAISS Embeddings with Financials over Fiscal Year 2019 to 2024 context
        Returns:
            FAISS: Embeddings
        """
        
        if os.path.exists("LocalEmbeddings/FinancialsOfCentralGovernment"):
            vectorstore = FAISS.load_local("LocalEmbeddings/FinancialsOfCentralGovernment", EmbeddingsDataModel.getEmbeddings(),allow_dangerous_deserialization=True)
            if not vectorstore:
                docs = self.convertTextToDocuments(FinancialsOfCentralGovernmentPath)

                vectorstore = FAISS.from_documents(docs, EmbeddingsDataModel.getEmbeddings())
                #store vector db 
                vectorstore.save_local(folder_path="LocalEmbeddings/FinancialsOfCentralGovernment")
        else:
            docs = self.convertTextToDocuments(FinancialsOfCentralGovernmentPath)
            # returns vector store db from documents and embeddings
            vectorstore = FAISS.from_documents(docs, EmbeddingsDataModel.getEmbeddings())
            #store vector db 
            vectorstore.save_local(folder_path="LocalEmbeddings/FinancialsOfCentralGovernment")
        # return The FAISS Embeddings Vector STore DB

        return vectorstore
    
    
        

    def getPandasData(DataPath):
        """_summary_

        Args:
            DataPath (str | Path): CSV file Path

        Returns:
            DataFrame: file to Dataframe
        """
        dataframe = pd.read_csv(DataPath)
        dataframe.dropna(inplace=True)
      
        #print(dataframe.head())
        return dataframe
        
    def getEmbeddings():
        """_summary_

        Return OllamaEmbeddings for the Models

        Returns:
            _type_: _description_
        """
        #considering openAI embeddings for ChatOpenAI
        embeddings = OllamaEmbeddings(model="gemma2:2b")
        return embeddings
    
    def getSubsidies(self):
        """_summary_
        Getter method for Subsidies


        Returns:
            DataFrame: _description_
        """
        SubsidiesCSV = EmbeddingsDataModel.getPandasData(SubsidiesDataPath)
        return SubsidiesCSV
    
    def getSpecialAllocationData(self):
        """_summary_
        Getter method for Special Allocations


        Returns:
            DataFrame: _description_
        """
        SpecialAllocationCSV = EmbeddingsDataModel.getPandasData(SpecialAllocationDataPath)
        return SpecialAllocationCSV

    
    def getCentralExpensesData(self):
        """_summary_
        Getter method for Central Expenses


        Returns:
            DataFrame: _description_
        """
        
        CentralExpensesData = EmbeddingsDataModel.getPandasData(CentralExpensesDataPath)
        return CentralExpensesData
    
    def getCentralBreakupData(self):
        """_summary_
        Getter method for Central Receipts


        Returns:
            DataFrame: _description_
        """
        CentralBreakupData=EmbeddingsDataModel.getPandasData(CentralBreakupDataPath)
        return CentralBreakupData
    
    def getBudgetGlanceData(self):
        """_summary_
        Getter method for Glance Over Budget Data


        Returns:
            DataFrame: _description_
        """
        BudgetGlanceData=EmbeddingsDataModel.getPandasData(BudgetGlanceDataPath)
        return BudgetGlanceData
    
    def getFRBMData(self):
        """_summary_
        Getter method for FRBM


        Returns:
            DataFrame: _description_
        """
        FRBMData=EmbeddingsDataModel.getPandasData(FRBMDataPath)
        return FRBMData
    
    def getMinistryData(self):
        """_summary_
        Getter method for Ministries Data


        Returns:
            DataFrame: _description_
        """
        MinistryData=EmbeddingsDataModel.getPandasData(MinistryDataPath)
        return MinistryData
    
    def getSchemeAllocationData(self):
        """_summary_
        Getter method for Scheme Based Allocations


        Returns:
            DataFrame: _description_
        """
        SchemeAllocationData=EmbeddingsDataModel.getPandasData(SchemeAllocationDataPath)
        return SchemeAllocationData
    
    def getTaxSlabsData(self):
        """_summary_
        Getter method for Tax Slabs

        Returns:
            DataFrame: _description_
        """
        TaxSlabsData=EmbeddingsDataModel.getPandasData(TaxSlabsDataPath)
        return TaxSlabsData

    

