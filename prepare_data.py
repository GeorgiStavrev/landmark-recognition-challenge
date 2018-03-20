import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO

def create_dir_structure():
    create_dir_if_not_exists('temp')
    create_dir_if_not_exists('train')

def create_dir_if_not_exists(dirname):
    directory = os.path.join(os.getcwd(), 'data', dirname)
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_file(id, url):
    r = requests.get(url, allow_redirects=True)
    img = Image.open(BytesIO(r.content))
    filepath = os.path.join(os.getcwd(), 'data', 'temp', id + '.' + img.format)

    img.thumbnail((224,224))
    img.save(filepath)

# HERE IT COMES
create_dir_structure()
print('Loading train.csv ...')
train_df = pd.read_csv('./data/train.csv')
print('Loaded train.csv')
print(train_df.head())

for index, row in train_df.iterrows():
    if index > 10:
        break

    download_file(row['id'], row['url'])
