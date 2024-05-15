from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

app = FastAPI()

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

#Data Ingestion
loader = PyPDFDirectoryLoader("./data_2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)

ollama_embeddings = OllamaEmbeddings()


#loading LLM
llm = ChatGroq(groq_api_key=groq_api_key, 
               model_name="Llama3-8b-8192")

#prompt
prompt = PromptTemplate.from_template("""
Act as helpfull Talent Transformation assistant and suggest the best Courses to take based on the given 
context, given {skills} and given {experience}. I want all the output in JSON format.

Prepare 3 sets of courses by total learning hours where Course type is Mandatory:
- Set 1 (1-day plan): Course duration < 3 hours
- Set 2 (3-day plan): Course duration between 3 and 20 hours
- Set 3 (5-day plan): Course duration > 20 hours

Also prepare 1 set of course where Course type is Optional that suits the {experience} and {skills}
- Optional set : Courses should be from Optional course type

For each suggested course, provide the following:
- Course Name
- Course Link
- Course Length
- Learner Level

Please ensure consistency in providing course details and consider the inclusion of mandatory courses in the recommendations.

Suggest the courses only based on the context knowledge
Keep your course recomandation ground in the facts of the context.
If the context does not contain the suitable courses according to the given {skills} then return "No course Found"

Be as concise as possible when giving answers. Do not anounce that you will answering. 

<context>
{context}
<context>

Skills: {skills}

Experience: {experience}
""")

# Define FastAPI endpoints
@app.get("/courses/")
async def get_courses(skills: str, experience: float):
    try:
        # Load documents
        docs = loader.load()
        
        # Split documents
        documents = text_splitter.split_documents(docs)
        
        # Create vector store
        db = FAISS.from_documents(documents, ollama_embeddings)
        
        # Retrieve courses
        retriever = db.as_retriever()
        setup = RunnableParallel(context=retriever, skills=RunnablePassthrough(), experience=RunnablePassthrough())
        chain = setup | prompt | llm
        response = chain.invoke({"skills": skills, "experience": experience})
        
        # Return course recommendations
        return response.content
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
