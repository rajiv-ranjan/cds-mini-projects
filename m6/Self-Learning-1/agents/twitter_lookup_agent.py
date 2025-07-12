from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from tools.tools import get_profile_url_tavily

load_dotenv()


def lookup(name: str, agent_llm) -> str:
    

    if agent_llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        llm = agent_llm

    template="""
       given the name {name_of_person} I want you to find a link to their Twitter/ X profile page, and extract from it their username
       In Your Final answer only the person's username
       which is extracted from: https://x.com/USERNAME"""
        
    prompt_template = PromptTemplate(
        input_variables=["name_of_person"],
        template=template,
    )

    tools_for_agent_twitter = [
        Tool(
            name="crawl Google for Twitter profile page",
            # Using a custom tool to search for LinkedIn profiles on Google
            # func=hub.pull("langchain/google-search-tool"),
            func=get_profile_url_tavily,  # Placeholder for the actual function to search LinkedIn profiles
            description="Use this tool to search for Twitter profiles and retrieve URL.",
        )
    ]

    # chain of thought prompt. Also called Reasoning and Action Prompt
    # https://smith.langchain.com/hub/hwchase17/react
    react_prompt = hub.pull("hwchase17/react")

    # Create the agent with the LLM, tools, and prompt
    agent = create_react_agent(llm=llm, tools=tools_for_agent_twitter, prompt=react_prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent_twitter, verbose=True)

    # Invoke the agent executor with the formatted prompt
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )
    print("Result:", result)

    twitter_user_name = result["output"]
    print(f"Tweeter username {name}: {twitter_user_name}")

    return twitter_user_name.strip()

