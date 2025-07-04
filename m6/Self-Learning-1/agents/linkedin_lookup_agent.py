import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from tools.tools import get_profile_url_tavily

load_dotenv()


def lookup(name: str, agent_llm) -> str:
    linked_profile_url = (
        "https://www.linkedin.com/in/eden-marco/"  # Default URL for testing purposes
    )

    if agent_llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        llm = agent_llm
        
    prompt_template = PromptTemplate(
        input_variables=["profile_name"],
        template="You are a helpful assistant. Look up the LinkedIn profile for {profile_name} and return the URL only.",
    )

    tools_for_agent = [
        Tool(
            name="crawl Google for LinkedIn profile page",
            # Using a custom tool to search for LinkedIn profiles on Google
            # func=hub.pull("langchain/google-search-tool"),
            func=get_profile_url_tavily,  # Placeholder for the actual function to search LinkedIn profiles
            description="Use this tool to search for LinkedIn profiles on Google and retrieve URL.",
        )
    ]

    # chain of thought prompt. Also called Reasoning and Action Prompt
    # https://smith.langchain.com/hub/hwchase17/react
    react_prompt = hub.pull("hwchase17/react")

    # Create the agent with the LLM, tools, and prompt
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    # Invoke the agent executor with the formatted prompt
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(profile_name=name)}
    )
    print("Result:", result)

    linked_profile_url = result["output"]
    print(f"LinkedIn profile URL for {name}: {linked_profile_url}")

    return linked_profile_url.strip()
