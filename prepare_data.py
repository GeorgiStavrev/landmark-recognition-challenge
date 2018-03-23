import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
import glob
from multiprocessing.dummy import Pool as ThreadPool
import sys

def download_batch_with_args(args):
    i, df, tag, img_side = args
    print('Starting Thread[{0}]'.format(i))
    download_batch(i, df, tag, img_side)

def get_checkpoint_filepath(thread_id, tag):
    filename = 'thread_' + tag + '_' + str(thread_id) + '.save'
    filepath = os.path.join(os.getcwd(), 'data', filename)
    return filepath

def write_checkpoint(thread_id, row_id, tag):
    with open(get_checkpoint_filepath(thread_id, tag), 'w') as text_file:
        print(str(row_id), file=text_file)

def read_checkpoint(thread_id, tag):
    filepath = get_checkpoint_filepath(thread_id, tag)
    if os.path.isfile(filepath):
        with open(filepath, 'r') as text_file:
            return True, text_file.read().rstrip()
    else:
        return False, ''


def download_batch(thread_id, df, tag, img_side):
    checkpoint_found, last_row_id = read_checkpoint(thread_id, tag)

    if checkpoint_found:
        print('Last row was \'' + last_row_id + '\'')
        position = df[df['id'] == last_row_id].index[0]
        print('Resuming thread ' + tag + '[' + str(thread_id) + '] from row', position)
        df = df.iloc[position:,:]

    errors = []
    for index, row in df.iterrows():
        url = row['url']
        img_id = row['id']
        if not download_resized_file(img_id, url, img_side):
            if not download_file(img_id, url, img_side):
                errors.append(img_id)

        if (index + 1)%100 == 0:
            write_checkpoint(thread_id, img_id, tag)

        if (index + 1)%10000 == 0:
            print('Processed {0} images so far.'.format(index))
    
    for i in range(len(errors)):
        row = df.loc[df['id'] == errors[i]]
        print('Test sample with id {0} was not imported. Url: {1}'.format(row['id'], row['url']))

def create_dir_structure():
    create_dir_if_not_exists('temp')

def create_dir_if_not_exists(dirname):
    directory = os.path.join(os.getcwd(), 'data', dirname)
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_resized_file(img_id, url, img_side):
    sizes = ['/s1600/','/w500/', '/s150-c/', '/s0-d/', '/d/', '/s0/']
    result = False
    for i in range(len(sizes)):
        if sizes[i] in url:
            new_url = url.replace(sizes[i], '/s' + str(img_side) + '/')
            if download_file(img_id, new_url, img_side):
                result = True
                break
    
    return result

def download_file(id, url, img_side):
    result = True
    if not file_exists_for_id(id):
        try:
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content))
                filepath = os.path.join(os.getcwd(), 'data', 'temp', id + '.' + img.format)
                img.thumbnail((img_side, img_side))
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
def download_test_data(parallel_threads_count, img_side):
    download_data('test', parallel_threads_count, img_side)

def download_train_data(parallel_threads_count, img_side):
    download_data('train', parallel_threads_count, img_side)

def download_data(name, parallel_threads_count, img_side):
    print('Loading ' + name + '.csv ...')
    data_path = os.path.join(os.getcwd(), 'data', name + '.csv')
    train_df = pd.read_csv(data_path)
    print('Loaded ' + name + '.csv')
    print(train_df.head())
    print(train_df.shape)

    threads_count = parallel_threads_count
    batch_size = train_df.shape[0] / threads_count
    download_batch_args = []
    for i in range(threads_count):
        from_idx = int(i * batch_size)
        to_idx = int((i+1) * batch_size)
        print('Add batch from {0} to {1}'.format(from_idx, to_idx))

        thread_args = (i, train_df.iloc[from_idx:to_idx,:], name + '_' + str(threads_count), img_side)
        download_batch_args.append(thread_args)

    pool = ThreadPool(threads_count)
    pool.map(download_batch_with_args, download_batch_args)
    pool.close()
    pool.join()

    print('Download ' + name + ' END')

if __name__ == '__main__':
    parallel_threads_count = 10
    args_count = len(sys.argv)
    task = ''
    img_side = 224

    if args_count > 1:
        task = sys.argv[1]

    if args_count > 2:
        parallel_threads_count = int(sys.argv[2])

    if args_count > 3:
        img_side = int(sys.argv[3])


    if args_count == 1 or task == 'test':
        create_dir_if_not_exists('temp')
        download_test_data(parallel_threads_count, img_side)
        os.rename(os.path.join(os.getcwd(), 'data', 'temp'), os.path.join(os.getcwd(), 'data', 'test'))

    if args_count == 1 or task == 'train':
        create_dir_if_not_exists('temp')
        download_train_data(parallel_threads_count, img_side)
        os.rename(os.path.join(os.getcwd(), 'data', 'temp'), os.path.join(os.getcwd(), 'data', 'train'))
