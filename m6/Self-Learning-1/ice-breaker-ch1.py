import os
import openai
import httpx
from dotenv import dotenv_values, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


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


def chat_with_AI(choice, chain_collection, information):
    chosen_chain = None
    if (choice is not None) and (choice in chain_collection):
        chosen_chain = chain_collection[choice]
    else:
        print("Invalid choice. Defaulting to OpenAI.")
        chosen_chain = chain_collection["O"]
    print(f"Using {chosen_chain.name} model for summarization.")
    print("Please wait while the AI processes the information...")
    try:
        res = chosen_chain.invoke(input={"information": information})
        print(res)
    except openai.RateLimitError as e:
        print("OpenAI API quota exceeded. Please check your plan and billing details.")
        print(e)
    except httpx.HTTPStatusError as e:
        print("HTTP error occurred:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)


def main():
    # read the documentation at https://pypi.org/project/python-dotenv/

    load_configs_and_secrets()

    print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")

    information = """
        Elon Reeve Musk (/ˈiːlɒn/; EE-lon; born June 28, 1971) is a businessman and investor. He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is the wealthiest person in the world, with an estimated net worth of US$232 billion as of December 2023, according to the Bloomberg Billionaires Index, and $254 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.[5][6]

A member of the wealthy South African Musk family, Elon was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University. However, Musk dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.

In October 2002, eBay acquired PayPal for $1.5 billion, and that same year, with $100 million of the money he made, Musk founded SpaceX, a spaceflight services company. In 2004, he became an early investor in electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. In 2006, Musk helped create SolarCity, a solar-energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year, Musk co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and the Boring Company, a tunnel construction company. In 2022, he acquired Twitter for $44 billion. He subsequently merged the company into newly created X Corp. and rebranded the service as X the following year. In March 2023, he founded xAI, an artificial intelligence company.

    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm_OpenAI = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0, max_tokens=None, max_retries=2
    )
    llm_Ollama_llama3 = ChatOllama(model="llama3", temperature=0)
    llm_Ollama_mistral = ChatOllama(model="mistral", temperature=0)
    # llm_Ollama = ChatOllama(temperature=0, model_name="llama3.1", base_url="http://localhost:11434")
    # llm_Ollama = ChatOllama(temperature=0, model_name="llama3.1", base_url="http://localhost:11434", request_timeout=60)

    chain_OpenAI = summary_prompt_template | llm_OpenAI | StrOutputParser()
    chain_Ollama_llama3 = (
        summary_prompt_template | llm_Ollama_llama3 | StrOutputParser()
    )
    chain_Ollama_mistral = (
        summary_prompt_template | llm_Ollama_mistral | StrOutputParser()
    )
    chain_collection = {
        "O": chain_OpenAI,
        "L": chain_Ollama_llama3,
        "M": chain_Ollama_mistral,
    }

    chat_with_AI_Choice = (
        input(
            "Which model do you want to invoke? (O for OpenAI, L for Ollama, M for Mistral): "
        )
        .strip()
        .upper()
    )
    chat_with_AI(chat_with_AI_Choice, chain_collection, information)


if __name__ == "__main__":
    main()
