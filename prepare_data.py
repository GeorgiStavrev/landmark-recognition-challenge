import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
import glob
from multiprocessing.dummy import Pool as ThreadPool

def download_batch(df):
    print('download_batch(df)')
    for index, row in df.iterrows():
        download_file(row['id'], row['url'])

def create_dir_structure():
    create_dir_if_not_exists('temp')
    create_dir_if_not_exists('train')

def create_dir_if_not_exists(dirname):
    directory = os.path.join(os.getcwd(), 'data', dirname)
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_file(id, url):
    if not file_exists_for_id(id):
        try:
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content))
                filepath = os.path.join(os.getcwd(), 'data', 'temp', id + '.' + img.format)

                img.thumbnail((224,224))
                img.save(filepath)
        except OSError:
            print('Something went wrong while trying to identify image with id:{0} and url:{0}'.format(id, url))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

def file_exists_for_id(id):
    res = False
    path = os.path.join(os.getcwd(), 'data', 'temp')
    for file in os.listdir(path):
        if file.startswith(id):
             res = True
             break

    return res

# HERE IT COMES
create_dir_structure()
print('Loading train.csv ...')
train_df = pd.read_csv('./data/train.csv')
print('Loaded train.csv')
print(train_df.head())
print(train_df.shape)

threads_count = 10
batch_size = train_df.shape[0] / threads_count
batches = []
for i in range(threads_count):
    from_idx = int(i * batch_size)
    to_idx = int((i+1) * batch_size)
    print('Add batch from {0} to {1}'.format(from_idx, to_idx))
    batches.append(train_df.iloc[from_idx:to_idx,:])

pool = ThreadPool(threads_count)
pool.map(download_batch, batches)
pool.close()
pool.join()

print('END')
