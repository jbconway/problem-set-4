'''
PART 2: SIMILAR ACTROS BY GENRE

Using the imbd_movies dataset:
- Create a data frame, where each row corresponds to an actor, each column represents a genre, and each cell captures how many times that row's actor has appeared in that column’s genre 
- Using this data frame as your “feature matrix”, select an actor (called your “query”) for whom you want to find the top 10 most similar actors based on the genres in which they’ve starred 
- - As an example, select the row from your data frame associated with Chris Hemsworth, actor ID “nm1165110”, as your “query” actor
- Use sklearn.metrics.DistanceMetric to calculate the euclidean distances between your query actor and all other actors based on their genre appearances
- - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
- Output a CSV continaing the top ten actors most similar to your query actor using cosine distance 
- - Name it 'similar_actors_genre_{current_datetime}.csv' to `/data`
- - For example, the top 10 for Chris Hemsworth are:  
        nm1165110 Chris Hemsworth
        nm0000129 Tom Cruise
        nm0147147 Henry Cavill
        nm0829032 Ray Stevenson
        nm5899377 Tiger Shroff
        nm1679372 Sudeep
        nm0003244 Jordi Mollà
        nm0636280 Richard Norton
        nm0607884 Mark Mortimer
        nm2018237 Taylor Kitsch
- Describe in a print() statement how this list changes based on Euclidean distance
- Make sure your code is in line with the standards we're using in this class
'''

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import DistanceMetric, pairwise_distances
from datetime import datetime

#Write your code below
def sag():
    """
    Finds the top 10 actors most similar to Chris Hemsworth based on genre appearances.
    Uses the JSON file from Part 1 ETL to build the actor × genre matrix.
    Outputs a CSV to /data with the results.
    """
    # Set up directories
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, "imdb_movies_2000to2022.prolific.json")

    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        movies = [json.loads(line) for line in f]

    # Build actor × genre feature matrix
    actor_genre_dict = {}  # actor_id -> {'name': actor_name, genre1: count, ...}

    for movie in movies:
        genres = movie.get('genres', [])
        actors = movie.get('actors', [])
        for actor_id, actor_name in actors:
            if actor_id not in actor_genre_dict:
                actor_genre_dict[actor_id] = {'name': actor_name}
            for genre in genres:
                actor_genre_dict[actor_id][genre] = actor_genre_dict[actor_id].get(genre, 0) + 1

    # Convert to DataFrame
    actor_genre_df = pd.DataFrame.from_dict(actor_genre_dict, orient='index').fillna(0)
    actor_genre_df = actor_genre_df.rename_axis('actor_id').reset_index()

    # Feature columns (all genres)
    feature_cols = [col for col in actor_genre_df.columns if col not in ['actor_id', 'name']]

    # Select query actor: Chris Hemsworth
    query_actor_id = 'nm1165110'
    query_features = actor_genre_df.loc[actor_genre_df['actor_id'] == query_actor_id, feature_cols].values
    all_features = actor_genre_df[feature_cols].values

    # Compute Euclidean distances
    dist = DistanceMetric.get_metric('euclidean')
    euclidean_distances = dist.pairwise(query_features, all_features).flatten()
    actor_genre_df['euclidean_distance'] = euclidean_distances

    # Compute cosine distances
    cosine_distances = pairwise_distances(query_features, all_features, metric='cosine').flatten()
    actor_genre_df['cosine_distance'] = cosine_distances
    
    # Top 10 actors by cosine distance (including the query actor)
    top_10 = actor_genre_df.sort_values('cosine_distance').head(10)

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(data_dir, f'similar_actors_genre_{timestamp}.csv')
    top_10[['actor_id', 'name', 'cosine_distance']].to_csv(output_csv, index=False)
    print(f"Top 10 similar actors (cosine distance) saved to {output_csv}")

    # Print top 10 by Euclidean distance
    top_10_euclid = actor_genre_df.sort_values('euclidean_distance').head(10)
    print("\nTop 10 actors by Euclidean distance:")
    for idx, row in top_10_euclid.iterrows():
        print(f"{row['actor_id']} {row['name']} (Euclidean distance: {row['euclidean_distance']:.3f})")
    print("\nThe ranking changes when using Euclidean distance because it looks at the overall difference in the number of movies actors have done in each genre, not just the pattern. So actors who act a lot in similar genres but at different frequencies may move up or down compared to the cosine distance ranking.")
