import os
import requests
from dotenv import load_dotenv
import json
import tweepy

load_dotenv()

twitter_client = tweepy.Client(
    bearer_token=os.environ["TWITTER_BEARER_TOKEN"],
    consumer_key=os.environ["TWITTER_API_KEY"],
    consumer_secret=os.environ["TWITTER_API_KEY_SECRET"],
    access_token=os.environ["TWITTER_ACCESS_TOKEN"],
    access_token_secret=os.environ["TWITTER_ACCESS_TOKEN_SECRET"],
)


def scrape_twitter_profile(username: str, num_of_tweets: int = 5, mock: bool = True):
    """
    Scrapes a Twitter user's original tweets (i.e., not retweets or replies) and returns them as a list of dictionaries.
    Each dictionary has three fields: "time_posted" (relative to now), "text", and "url".
    """

    tweet_list = []
    tweets=[]

    if mock:
        # Mock response has 2 tweets
        twitter_profile_url = "https://gist.githubusercontent.com/emarco177/827323bb599553d0f0e662da07b9ff68/raw/57bf38cf8acce0c87e060f9bb51f6ab72098fbd6/eden-marco-twitter.json"

        # Mock response has more than 20 tweets
        # twitter_profile_url = "https://gist.githubusercontent.com/rajiv-ranjan/f9d5c702cf5905f1c440d9b76136eda3/raw/a743fd9b3f55853e56422121bff2dc61c3ce659c/tweet-mock.json"

        response = requests.get(
            twitter_profile_url,
            timeout=10,
        )

        tweets = response.json()
        for tweet in tweets:
            tweet_dict = {}
            tweet_dict["text"] = tweet["text"]
            tweet_dict["url"] = f"https://twitter.com/{username}/status/{tweet['id']}"
            tweet_list.append(tweet_dict)
    else:
        user_id = twitter_client.get_user(username=username).data.id
        # Raw Response that I pulled from twitter API
        # Response(data=[<Tweet id=1936687115611394130 text='Cursor is down. What do I do now?'>, <Tweet id=1934876841178788287 text='sweet learning repo for productionizing ur ai apps.\n\ngreat job @NirDiamantAI \nhttps://t.co/VeQSEUr5Xo https://t.co/nUPTybw31c'>, <Tweet id=1934680466709225606 text='This is cool https://t.co/obMszxdBAz'>, <Tweet id=1933535567393726935 text='Just because you can ship 100x faster today doesn’t mean you should.\n\nThe worst thing you can do is build something nobody wants.@assaf_elovic &amp; @hwchase17 brilliantly articulate \nCAIR:\n Confidence = Value ÷ (Risk × Correction Effort\n\na metric that determines real AI product https://t.co/EMPLuOzruX'>, <Tweet id=1929194846591357106 text="I literally forgot how to spell, i don't even try these days.\nUse LLMs way too much that I cant spell for my life 😅\n\nIs this like using a printed maps for navigation or something I should address? (post written 100% by me)">, <Tweet id=1926891802298102262 text='→ Use .cursorrules \n→ "Write tests before coding"\n\nYou can thank me later'>, <Tweet id=1924880134407921936 text='Exposing LangGraph graphs as MCP servers feels very natural and will def bring tons of innovation https://t.co/xrC5c5wvzj'>, <Tweet id=1924879895420669990 text='been waiting for this in a while!!! https://t.co/kxXWhCBMub'>, <Tweet id=1923115383390863485 text="Had a blast at @LangChainAI  in SF! 🙌\n\nLearning about all the innovations is cool, but the real magic? Conencting with the incredible #LangChain commnuity.\n \nSeriously, the strength &amp; energy of this open source group is unmatched. Haven't seen anything like it. \n\nTHAT's the true https://t.co/OP2D9VEvOw">, <Tweet id=1923046575154098198 text='Scrolling through old code:\n“Bruh, who tf wrote this garbage?!”\n👇\ngit blame\n👀\n…it was me. 2 years ago \n(Before @cursor_ai ) wonder what its gonna be like 2 years forward'>, <Tweet id=1922803372807086369 text="Learning from @assaf_elovic's session at @LangChainAI interrupt.\n\nAgents in production are here🤖\n\nImpressive to see what @mondaydotcom are up to and  examples how to take agentic workloads to production. https://t.co/1qp59UX9Rd">, <Tweet id=1922775305678111059 text="Learning from @Uber 's fantastic ai journey implementing multi agents at @LangChainAI interrupt.\n\nThis is THE place for intaking  lesson learned in production grade ai apps. https://t.co/sHn6AlAWr9">, <Tweet id=1917981258325319905 text='MCP on the frontend s a paradigm shift with real-world impact. Remember, you don’t need a sledgehammer to drive a nail 🔨\n\nI am predicting that this repo by @CopilotKit 🪁 will go viral https://t.co/pn6C4dxAc6'>, <Tweet id=1915464608954929176 text='Anyone wants to arrange competitive vibing? \n\nHow this sport is going to look like?😎'>, <Tweet id=1914937602488025197 text='## Markdown is quietly becoming the lingua franca of AI agents and vibe coders\n\nNot JSON. Not XML.  \nJust clean, readable, flexible **Markdown**. https://t.co/eQMnwwYWBA'>, <Tweet id=1914761465854415330 text='Think of compiling a Certified Professional Vibe Coder exam 😂\n\nAt this point its only a matter of time before cloud providers start offering \n "Certified Professional Vibe Coder" \n\nWhat  questions should be on the exam? @cursor_ai @windsurf_ai @v0 @lovable_dev @boltdotnew https://t.co/GXLH1W1Y9A'>, <Tweet id=1914487120195887408 text='What makes someone unstoppable today isn’t deep expertise in one thing. \nit’s being dangerous enough in many.\nGeneralists who can move fast, connect ideas, and ship with AI are about to change everything. https://t.co/1OJgaU09ZI'>, <Tweet id=1912251738347958671 text='100% https://t.co/xKfrnZpzDv'>, <Tweet id=1911633936679129303 text='Everyone’s a dev now. PMs, marketers, you name it.\nTools like @cursor_ai , @windsurf_ai , and @lovable_dev  have blurred the line and commoditized software.\n\nBut devs? We’re crossing lines too. becoming PMs, marketers, storytellers.\n\nNo better time to be a developer.\nNo worse https://t.co/lSfRFY7tdq'>], includes={}, errors=[], meta={'result_count': 19, 'newest_id': '1936687115611394130', 'oldest_id': '1911633936679129303', 'next_token': '7140dibdnow9c7btw4e02bdaohrkc9n13j167of4x5iyb'})
        try:
            tweets = twitter_client.get_users_tweets(
                id=user_id, max_results=num_of_tweets, exclude=["retweets", "replies"]
            )
            tweets = tweets.data if tweets.data else []
        except Exception as e:
            print(f"Error fetching tweets for {username}: {e}")
            
        for tweet in tweets:
            tweet_dict = {}
            tweet_dict["text"] = tweet["text"]
            tweet_dict["url"] = f"https://twitter.com/{username}/status/{tweet.id}"
            tweet_list.append(tweet_dict)

    return tweet_list


if __name__ == "__main__":
    username = "EdenEmarco177"
    tweet_collection = scrape_twitter_profile(
        username=username, num_of_tweets=20, mock=True
    )
    print(json.dumps(tweet_collection, indent=2, ensure_ascii=False))



## tweet_collection values when mock is set to False
# [
#   {
#     "text": "Cursor is down. What do I do now?",
#     "url": "https://twitter.com/EdenEmarco177/status/1936687115611394130"
#   },
#   {
#     "text": "sweet learning repo for productionizing ur ai apps.\n\ngreat job @NirDiamantAI \nhttps://t.co/VeQSEUr5Xo https://t.co/nUPTybw31c",
#     "url": "https://twitter.com/EdenEmarco177/status/1934876841178788287"
#   },
#   {
#     "text": "This is cool https://t.co/obMszxdBAz",
#     "url": "https://twitter.com/EdenEmarco177/status/1934680466709225606"
#   },
#   {
#     "text": "Just because you can ship 100x faster today doesn’t mean you should.\n\nThe worst thing you can do is build something nobody wants.@assaf_elovic &amp; @hwchase17 brilliantly articulate \nCAIR:\n Confidence = Value ÷ (Risk × Correction Effort\n\na metric that determines real AI product https://t.co/EMPLuOzruX",
#     "url": "https://twitter.com/EdenEmarco177/status/1933535567393726935"
#   },
#   {
#     "text": "I literally forgot how to spell, i don't even try these days.\nUse LLMs way too much that I cant spell for my life 😅\n\nIs this like using a printed maps for navigation or something I should address? (post written 100% by me)",
#     "url": "https://twitter.com/EdenEmarco177/status/1929194846591357106"
#   },
#   {
#     "text": "→ Use .cursorrules \n→ \"Write tests before coding\"\n\nYou can thank me later",
#     "url": "https://twitter.com/EdenEmarco177/status/1926891802298102262"
#   },
#   {
#     "text": "Exposing LangGraph graphs as MCP servers feels very natural and will def bring tons of innovation https://t.co/xrC5c5wvzj",
#     "url": "https://twitter.com/EdenEmarco177/status/1924880134407921936"
#   },
#   {
#     "text": "been waiting for this in a while!!! https://t.co/kxXWhCBMub",
#     "url": "https://twitter.com/EdenEmarco177/status/1924879895420669990"
#   },
#   {
#     "text": "Had a blast at @LangChainAI  in SF! 🙌\n\nLearning about all the innovations is cool, but the real magic? Conencting with the incredible #LangChain commnuity.\n \nSeriously, the strength &amp; energy of this open source group is unmatched. Haven't seen anything like it. \n\nTHAT's the true https://t.co/OP2D9VEvOw",
#     "url": "https://twitter.com/EdenEmarco177/status/1923115383390863485"
#   },
#   {
#     "text": "Scrolling through old code:\n“Bruh, who tf wrote this garbage?!”\n👇\ngit blame\n👀\n…it was me. 2 years ago \n(Before @cursor_ai ) wonder what its gonna be like 2 years forward",
#     "url": "https://twitter.com/EdenEmarco177/status/1923046575154098198"
#   },
#   {
#     "text": "Learning from @assaf_elovic's session at @LangChainAI interrupt.\n\nAgents in production are here🤖\n\nImpressive to see what @mondaydotcom are up to and  examples how to take agentic workloads to production. https://t.co/1qp59UX9Rd",
#     "url": "https://twitter.com/EdenEmarco177/status/1922803372807086369"
#   },
#   {
#     "text": "Learning from @Uber 's fantastic ai journey implementing multi agents at @LangChainAI interrupt.\n\nThis is THE place for intaking  lesson learned in production grade ai apps. https://t.co/sHn6AlAWr9",
#     "url": "https://twitter.com/EdenEmarco177/status/1922775305678111059"
#   },
#   {
#     "text": "MCP on the frontend s a paradigm shift with real-world impact. Remember, you don’t need a sledgehammer to drive a nail 🔨\n\nI am predicting that this repo by @CopilotKit 🪁 will go viral https://t.co/pn6C4dxAc6",
#     "url": "https://twitter.com/EdenEmarco177/status/1917981258325319905"
#   },
#   {
#     "text": "Anyone wants to arrange competitive vibing? \n\nHow this sport is going to look like?😎",
#     "url": "https://twitter.com/EdenEmarco177/status/1915464608954929176"
#   },
#   {
#     "text": "## Markdown is quietly becoming the lingua franca of AI agents and vibe coders\n\nNot JSON. Not XML.  \nJust clean, readable, flexible **Markdown**. https://t.co/eQMnwwYWBA",
#     "url": "https://twitter.com/EdenEmarco177/status/1914937602488025197"
#   },
#   {
#     "text": "Think of compiling a Certified Professional Vibe Coder exam 😂\n\nAt this point its only a matter of time before cloud providers start offering \n \"Certified Professional Vibe Coder\" \n\nWhat  questions should be on the exam? @cursor_ai @windsurf_ai @v0 @lovable_dev @boltdotnew https://t.co/GXLH1W1Y9A",
#     "url": "https://twitter.com/EdenEmarco177/status/1914761465854415330"
#   },
#   {
#     "text": "What makes someone unstoppable today isn’t deep expertise in one thing. \nit’s being dangerous enough in many.\nGeneralists who can move fast, connect ideas, and ship with AI are about to change everything. https://t.co/1OJgaU09ZI",
#     "url": "https://twitter.com/EdenEmarco177/status/1914487120195887408"
#   },
#   {
#     "text": "100% https://t.co/xKfrnZpzDv",
#     "url": "https://twitter.com/EdenEmarco177/status/1912251738347958671"
#   },
#   {
#     "text": "Everyone’s a dev now. PMs, marketers, you name it.\nTools like @cursor_ai , @windsurf_ai , and @lovable_dev  have blurred the line and commoditized software.\n\nBut devs? We’re crossing lines too. becoming PMs, marketers, storytellers.\n\nNo better time to be a developer.\nNo worse https://t.co/lSfRFY7tdq",
#     "url": "https://twitter.com/EdenEmarco177/status/1911633936679129303"
#   }
# ]