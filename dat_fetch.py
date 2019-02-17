import requests
import shutil
import pandas as pd
import json
import os

def downloader(image_url, file_name):
    r = requests.get(image_url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
    if r.status_code == 200:
        with open(file_name + '.png', 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


url_df = pd.read_csv('anime_em.csv')
prefix = 'em_%i'
url_list = url_df['imgs-src']

counter = 0

for url in url_list:
    downloader(url, prefix % (counter))
    print(counter)
    counter += 1
