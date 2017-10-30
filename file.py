import glob
import os
import shutil

def split_dir(train_dir, n=2000):
    '''
    Moves random sample of files to valid directory.

    ex) split_dir('~/data/train')
    '''

    if train_dir[0] == '~':
        train_dir = os.path.expanduser(train_dir)

    valid_dir = os.path.normpath(os.path.join(train_dir, '../valid'))
    os.makedirs(valid_dir)

    files = glob.glob(os.path.join(train_dir, '*.jpg'))
    shuffled = np.random.permutation(files)

    n = min(n, len(shuffled))

    for i in range(n):
        os.rename(shuffled[i], os.path.join(valid_dir, os.path.basename(shuffled[i])))

def sample_dir(train_dir, n=2000):
    '''
    Copies random sample of files to sample directory.

    ex) sample_dir('~/data/train')
    '''

    if train_dir[0] == '~':
        train_dir = os.path.expanduser(train_dir)

    sample_dir = os.path.normpath(os.path.join(train_dir, '../sample'))
    os.makedirs(sample_dir)

    files = glob.glob(os.path.join(train_dir, '*.jpg'))
    shuffled = np.random.permutation(files)

    n = min(n, len(shuffled))

    for i in range(n):
        shutil.copyfile(shuffled[i], os.path.join(valid_dir, os.path.basename(shuffled[i])))

def create_label_dir(folder, categories):
    '''
    Moves each file to its own directory based on its name.

    ex) create_label_dir('~/data/train', ['cat', 'dog'])
    '''

    if folder[0] == '~':
        folder = os.path.expanduser(folder)

    for label in categories:
        label_dir = os.path.join(folder, label)
        os.makedirs(label_dir)

        files = glob.glob(os.path.join(folder, '%s*.txt' % label))
        for f in files:
            shutil.move(f, os.path.join(label_dir, os.path.basename(f)))
#####

def create_dir(folder):
    try:
        os.makedirs(folder)
    except OSError:
        if not os.path.isdir(folder):
            raise

def create_pred_csv(model, directory, batch_size):
    unknown_folder = os.path.join(directory, 'unknown')
    create_dir(unknown_folder)

    files = os.listdir(unknown_folder)
    num_files = len(files)

    generator = image.ImageDataGenerator()
    batches = generator.flow_from_directory(directory, target_size=(224,224), shuffle=False, class_mode=None,
                                            batch_size=batch_size)
    filenames = batches.filenames

    with open('predictions.csv', "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(('id', 'label'))

        for i, batch in enumerate(batches):
            if i >= num_files / batch_size:
                break

            ids = [name.split('/')[1].split('.')[0] for name in filenames[batch_size*i: batch_size*(i+1)]]
            probs, indices, labels = model.predict(batch)
            preds = [a if b == 1 else 1 - a for a, b in zip(probs, indices)]
            data = zip(ids, preds)
            writer.writerows(data)

            num_finished = (i+1)*batch_size
            if num_finished > num_files:
                print('Finished batch %s: %s images' % (i+1, num_files))
            else:
                print('Finished batch %s: %s images' % (i+1, num_finished))
