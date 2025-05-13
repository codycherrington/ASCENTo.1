import praw
import json

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id="sBPGhV2ioYr1PQx5M22eXg",
    client_secret="RTDoqkzdgH-1nTiXFRNei0DDkzGlgg",
    user_agent="AscentScraper by u/Electronic-Worry-196"
)

# Subreddits to scrape
subreddits = ["CasualConversation", "AskReddit", "offmychest"]
limit = 1000  # Number of posts per subreddit (increased from 100 to 1000)

conversation_pairs = []

def extract_conversations_from_comments(comments):
    comments.replace_more(limit=0)
    comment_list = comments.list()
    for i in range(len(comment_list) - 1):
        parent = comment_list[i]
        child = comment_list[i + 1]
        if child.parent_id == parent.fullname:
            convo = {
                "input": parent.body.strip(),
                "output": child.body.strip()
            }
            # Filter out short or non-conversational comments
            if len(convo["input"]) > 5 and len(convo["output"]) > 5:
                conversation_pairs.append(convo)

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    print(f"Scraping subreddit: {subreddit_name}")
    for submission in subreddit.top(time_filter="all", limit=limit):
        submission.comments.replace_more(limit=0)
        extract_conversations_from_comments(submission.comments)

with open("ascent_data/reddit_conversations.json", "w") as f:
    json.dump(conversation_pairs, f, indent=2)

print(f"✅ Scraped {len(conversation_pairs)} conversation pairs and saved to 'ascent_data/reddit_conversations.json'")
