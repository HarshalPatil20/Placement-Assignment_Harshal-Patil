

from googleapiclient.discovery import build
import pandas as pd

# Set up YouTube Data API client
api_key = 'your_youtube_api_key'
youtube = build('youtube', 'v3', developerKey=api_key)

# Define video ID (extracted from the YouTube video link)
video_id = 'your_video_id'

# Retrieve comments using YouTube Data API
def get_video_comments(youtube, **kwargs):
    comments = []
    response = youtube.commentThreads().list(**kwargs).execute()

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        if 'nextPageToken' in response:
            kwargs['pageToken'] = response['nextPageToken']
            response = youtube.commentThreads().list(**kwargs).execute()
        else:
            break

    return comments

# Extract comments from the video
comments = get_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')

# Create a DataFrame and store comments in a CSV file
df = pd.DataFrame(comments, columns=['Comment'])
df.to_csv('video_comments.csv', index=False)
