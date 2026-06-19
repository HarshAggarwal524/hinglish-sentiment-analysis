import pandas as pd


def load_conll(filepath):
    tweets = []

    current_id = None
    current_sentiment = None
    current_tokens = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Empty line = end of tweet
            if not line:
                continue

            parts = line.split("\t")

            # New tweet starts
            if parts[0] == "meta":
                if current_id is not None:
                    tweets.append({
                        "tweet_id": current_id,
                        "sentiment": current_sentiment,
                        "text": " ".join(current_tokens)
                    })

                current_id = parts[1]
                current_sentiment = parts[2]
                current_tokens = []

            else:
                token = parts[0]
                current_tokens.append(token)

    # Add last tweet
    if current_id is not None:
        tweets.append({
            "tweet_id": current_id,
            "sentiment": current_sentiment,
            "text": " ".join(current_tokens)
        })

    return pd.DataFrame(tweets)