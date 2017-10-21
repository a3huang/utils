import os
import shutil

def split_dir(directory, categories):
    files = os.listdir(directory)

    for cat in categories:
        cat_dir = os.path.join(directory, cat)
        os.makedirs(cat_dir)
        print 'folder created: %s' % cat_dir

        cat_files = [i for i in os.listdir(directory) if cat in i]
        for f in cat_files:
            shutil.move(os.path.join(directory, f), cat_dir)
        print 'files moved to folder: %s' % cat_dir

def train_valid_split(directory):
    train_dir = os.path.join(directory, 'train')
    valid_dir = os.path.join(directory, 'valid')
    os.makedirs(valid_dir)

    for cls in next(os.walk(train_dir))[1]:
        cls_dir = os.path.join(train_dir, cls)

        files = os.listdir(cls_dir)
        random.shuffle(files)
        valid_files = files[:int(len(files)*0.3)]
        valid_cls_dir = os.path.join(valid_dir, cls)
        os.makedirs(valid_cls_dir)
        print 'folder created: %s' % valid_cls_dir

        for f in valid_files:
            shutil.move(os.path.join(cls_dir, f), valid_cls_dir)
        print 'files moved to folder: %s' % valid_cls_dir

def sample_dir(directory):
    parts = directory.split('/')
    sample_dir = os.path.join(parts[0], 'sample', parts[1])
    os.makedirs(sample_dir)

    for cls in next(os.walk(directory))[1]:
        cls_dir = os.path.join(directory, cls)

        files = os.listdir(cls_dir)
        random.shuffle(files)
        sample_files = files[:100]

        sample_cls_dir = os.path.join(sample_dir, cls)
        os.makedirs(sample_cls_dir)
        print 'folder created: %s' % sample_cls_dir

        for f in sample_files:
            shutil.copy(os.path.join(cls_dir, f), sample_cls_dir)
        print 'files copied to folder: %s' % sample_cls_dir

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

def create_sample_dir(directory, n=500):
    root = directory.split('/')[0]
    files = os.listdir(directory)
    random.shuffle(files)

    sample_files = files[:n]
    sample_cls_dir = os.path.join(root, 'test_copy')
    create_dir(sample_cls_dir)

    for f in sample_files:
        shutil.copy(os.path.join(directory, f), sample_cls_dir)

def move_to_dir(directory):
    new_dir = os.path.join(directory, 'unknown')
    create_dir(new_dir)

    if os.listdir(new_dir) == []:
        files = [i for i in os.listdir(directory) if os.path.isfile(os.path.join(directory, i))

        for f in files:
            shutil.move(f, new_dir)
