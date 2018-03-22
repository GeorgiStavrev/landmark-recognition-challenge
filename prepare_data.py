import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
import glob
from multiprocessing.dummy import Pool as ThreadPool
import sys
#from urllib import request, error

def download_batch(df):
    errors = []
    i = 0
    for index, row in df.iterrows():
        if not download_file(row['id'], row['url']):
            errors.append(row['id'])

        if (index + 1)%10000 == 0:
            print('Processed {0} images so far.'.format(index))')
    
    for i in len(errors):
        row = df.loc[df['id'] == errors[i]]
        print('Test sample with id {0} was not imported. Url: {1}'.format(row['id'], row['url']))

def create_dir_structure():
    create_dir_if_not_exists('temp')

def create_dir_if_not_exists(dirname):
    directory = os.path.join(os.getcwd(), 'data', dirname)
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_file(id, url):
    result = True
    if not file_exists_for_id(id):
        try:
            #response = request.urlopen(url)
            #image_data = response.read()
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content))
                filepath = os.path.join(os.getcwd(), 'data', 'temp', id + '.' + img.format)

                img.thumbnail((224,224))
                img.save(filepath)
        except OSError:
            print('Something went wrong while trying to identify image with id:{0} and url:{1}'.format(id, url))
            result = False
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    return result

def file_exists_for_id(id):
    res = False
    path = os.path.join(os.getcwd(), 'data', 'temp')
    for file in os.listdir(path):
        if file.startswith(id):
             res = True
             break

    return res

# HERE IT COMES
def download_test_data(parallel_threads_count):
    download_data('test', parallel_threads_count)

def download_train_data(parallel_threads_count):
    download_data('train', parallel_threads_count)

def download_data(name, parallel_threads_count):
    print('Loading ' + name + '.csv ...')
    data_path = os.path.join(os.getcwd(), 'data', name + '.csv')
    train_df = pd.read_csv(data_path)
    print('Loaded ' + name + '.csv')
    print(train_df.head())
    print(train_df.shape)

    threads_count = parallel_threads_count
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

    print('Download ' + name + ' END')

if __name__ == '__main__':
    args = sys.argv
    parallel_threads_count = 10
    args_count = len(sys.argv)

    if args_count > 2:
        parallel_threads_count = int(sys.argv[2])

    if args_count == 1 or sys.argv[1] == 'test':
        create_dir_if_not_exists('temp')
        download_test_data(parallel_threads_count)
        os.rename(os.path.join(os.getcwd(), 'data', 'temp'), os.path.join(os.getcwd(), 'data', 'test'))

    if args_count == 1 or sys.argv[1] == 'train':
        create_dir_if_not_exists('temp')
        download_train_data(parallel_threads_count)
        os.rename(os.path.join(os.getcwd(), 'data', 'temp'), os.path.join(os.getcwd(), 'data', 'train'))
