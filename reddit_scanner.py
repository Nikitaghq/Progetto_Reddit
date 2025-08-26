import praw
import os
import time
import json
from datetime import datetime
from tqdm import tqdm  

# API Reddit Config
reddit = praw.Reddit(
    client_id="",                            # Parametri cancellati per privacy
    client_secret="",                        # Parametri cancellati per privacy 
    user_agent="RedditDataCollector/1.0"
)

OUTPUT_DIR = "reddit_data" # Directory di output
LIMIT = 5000  # Numero di post da raccogliere

# Funzione per il download dei posts
def scrape_reddit():
    posts = []
    
    # Progeress Bar
    pbar = tqdm(total=LIMIT, desc="Scraping Reddit", unit="post")   
    try:
        for post in reddit.subreddit("all").hot(limit=LIMIT):
            try:
                created_dt = datetime.fromtimestamp(post.created_utc)
                post_hint = getattr(post, "post_hint", None)
                if post_hint == "image":
                    content_type = "image"
                elif post_hint in ("hosted:video", "rich:video") or post.is_video:
                    content_type = "video"
                elif post_hint == "link":
                    content_type = "link"
                elif post_hint == "gallery":
                    content_type = "gallery"
                elif post.is_self:
                    content_type = "text"
                else:
                    content_type = "unknown"

                post_data = {
                    "subreddit": post.subreddit.display_name,
                    "title": post.title,
                    "text": post.selftext[:500] if post.selftext else "",
                    "url": post.url,
                    "created_utc": created_dt.isoformat(),
                    "hour": created_dt.hour,
                    "weekday": created_dt.weekday(),


                    "score": post.score,
                    "num_comments": post.num_comments,
                    "upvote_ratio": post.upvote_ratio,
                    "awards": post.total_awards_received,


                    "is_original_content": post.is_original_content,
                    "spoiler": post.spoiler,
                    "nsfw": post.over_18,
                    "stickied": post.stickied,
                    "locked": post.locked,
                    "edited": bool(post.edited),
                    "distinguished": post.distinguished or "none",


                    "content_type": content_type,


                    "author_link_karma": getattr(post.author, "link_karma", None) if post.author else None,
                    "author_comment_karma": getattr(post.author, "comment_karma", None) if post.author else None
                }

                posts.append(post_data)
                
            except Exception as e:
                print(f"\n- Errore su post {getattr(post, 'id', 'sconosciuto')}: {e}")
            finally:
                pbar.update(1)
                time.sleep(1.5)  # Limiti imposti da Reddit, non cambiare
    finally:
        pbar.close() 

    return posts

os.makedirs(OUTPUT_DIR, exist_ok=True)
posts = scrape_reddit()

# Salva tutto in json
json_path = os.path.join(OUTPUT_DIR, "reddit_viral_posts.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(posts, f, indent=2, ensure_ascii=False)
print(f"\n- File JSON salvato: {json_path} ({len(posts)} post)")