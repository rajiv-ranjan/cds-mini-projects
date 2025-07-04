import os
import openai
import httpx
from dotenv import dotenv_values, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

def load_configs_and_secrets():
    # load_dotenv(dotenv_path=".env-secret")
    # load_dotenv(dotenv_path=".env-shared")
    # load_dotenv(dotenv_path=".env")
    load_dotenv(dotenv_path=".env")

    config = {
        **dotenv_values(".env"),  # load sensitive variables
        **dotenv_values(".env-shared"),  # load shared variables
        # **os.environ,  # override loaded values with environment variables
    }
    return config

def get_llms(choice:str) -> dict:
    # the models that we can use for the linkedin scrapping task are: https://langchaincourse.notion.site/1a342623556380c4baa3daba95f6e34f?v=1a3426235563803cbca7000c39ebfa23

    llm_collection = {}
    llm_OpenAI = ChatOpenAI(
        # model="gpt-3.5-turbo", 
        model="gpt-4o-mini", 
        temperature=0, 
        max_tokens=None, 
        max_retries=2
    )
    llm_collection["O"] = llm_OpenAI
    llm_Ollama_llama3 = ChatOllama(model="llama3", temperature=0)
    llm_collection["L"] = llm_Ollama_llama3
    llm_Ollama_mistral = ChatOllama(model="mistral", temperature=0)
    llm_collection["M"] = llm_Ollama_mistral
    # llm_GoogleGemini = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash",
    #     temperature=0)
    # llm_collection["G"] = llm_GoogleGemini

    # TODO : call Deepseek api & google gemini api
    # llm_Ollama = ChatOllama(temperature=0, model_name="llama3.1", base_url="http://localhost:11434")
    # llm_Ollama = ChatOllama(temperature=0, model_name="llama3.1", base_url="http://localhost:11434", request_timeout=60)    

    
    llm = llm_collection[choice] if choice in llm_collection else llm_collection["O"]

    print(f"You have selected the {llm.name} model for LinkedIn profile scraping.")
    
    return llm


def ice_break_with(name: str,llm) -> str:
    linkedin_url = linkedin_lookup_agent(name=name,agent_llm=llm)
    # mock is set to True as I do not have a valid Scrapin API key
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_url, mock=True)

    summary_template = """
    given the Linkedin information {linkedin_information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information"], template=summary_template
    )

    summary_chain = summary_prompt_template | llm 

    print(f"Using {llm.name} model for summarization.")
    print("Please wait while the AI processes the information...")
    try:
        res = summary_chain.invoke(input={"linkedin_information": linkedin_data})
        print(res)
    except openai.RateLimitError as e:
        print("OpenAI API quota exceeded. Please check your plan and billing details.")
        print(e)
    except httpx.HTTPStatusError as e:
        print("HTTP error occurred:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)

    return summary_prompt_template

def main():
    # read the documentation at https://pypi.org/project/python-dotenv/

    load_configs_and_secrets()
    choice_of_llm = (
        input(
            "Which model do you want to invoke? (O for OpenAI, L for Ollama, M for Mistral, G for Google Gemini): "
        )
        .strip()
        .upper()
    )
    llm = get_llms(choice_of_llm)

    ice_break_with(name="Eden Marco Udemy",llm=llm)

    

    


if __name__ == "__main__":
    main()




