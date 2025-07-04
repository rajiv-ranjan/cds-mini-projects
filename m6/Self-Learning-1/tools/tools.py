# https://www.tavily.com/
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

def get_profile_url_tavily(name: str) -> str:
    """Use Tavily to search for LinkedIn or Twitter profiles and return the URL."""
    
    search = TavilySearchResults()
    res = search.run(f"{name}")

    # # Initialize Tavily client with your API key
    # client = TavilySearchResults(api_key="YOUR_TAVILY_API_KEY")
    
    # # Search for the profile
    # response = client.search(query=name + " LinkedIn")
    
    # # Extract the first result's URL
    # if response and response.results:
    #     return response.results[0].url
    
    return res


# sample code from the tavily portal
# To install: pip install tavily-python

# from tavily import TavilyClient
# client = TavilyClient("")
# response = client.search(
#     query="Eden Marco Udemy LinkedIn"
# )
# print(response)