'''
PART 1: ETL the dataset and save in `data/`

Here is the imbd_movie data:
https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true

It is in JSON format, so you'll need to handle accordingly and also figure out what's the best format for the two analysis parts. 
'''

import os
import pandas as pd
import json

def etl():
    """
    Downloads the IMDb movies dataset (2000–2022), saves it as JSON 
    and CSV in '/data' for further analysis.

    Outputs:
        - /data/imdb_movies_2000to2022.prolific.json
        - /data/imdb_movies_2000to2022.csv
    """
    # Create '/data' directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Load datasets and save to '/data'
    imdb_movie_url = "https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true"

    # Load JSON directly into DataFrame
    df = pd.read_json(imdb_movie_url, lines=True)
    # Save JSON locally as well
    json_path = os.path.join(data_dir, "imdb_movies_2000to2022.prolific.json")
    df.to_json(json_path, orient="records", lines=True)
    print(f"Saved JSON to {json_path}")

    # Save DataFrame as CSV in /data
    csv_path = os.path.join(data_dir, "imdb_movies_2000to2022.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV to {csv_path}")