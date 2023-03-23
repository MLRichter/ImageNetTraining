from os.path import exists, join, curdir
from os import listdir, mkdir, sep
from shutil import rmtree, copyfile
from tqdm import tqdm

ROOT = join("..", "food-101", "food-101")
IMG_ROOT = join(ROOT, "images")
TGT_ROOT_TRAIN = join(ROOT, "train")
TGT_ROOT_TEST = join(ROOT, "test")

with open(join(ROOT, "meta", 'train.txt'), 'r') as fp:
    train_files = fp.readlines()
    train_files = [file.replace('\n', '') for file in train_files]

print("found {} train files".format(len(train_files)))

with open(join(ROOT, "meta", 'test.txt'), 'r') as fp:
    test_files = fp.readlines()
    test_files = [file.replace('\n', '') for file in test_files]

print("found {} test files".format(len(test_files)))

def setup_goal_structure(root, target):
    if not exists(target):
        print("Creating target foldet at {}".format(target))
        mkdir(target)
    print("Checking if all {} class folders exist".format(len(listdir(root))))
    for cls in listdir(root):
        if not exists(join(target, cls)):
            mkdir(join(target, cls))
            print("Creating class folder for {}".format(cls))

print('File structure setup startet')
setup_goal_structure(IMG_ROOT, TGT_ROOT_TRAIN)
setup_goal_structure(IMG_ROOT, TGT_ROOT_TEST)
print('File structure setup complete')

def copy_to_tagret(root, target, filelist):
    for i, file in enumerate(tqdm(filelist)):
        copyfile(join(root, file).replace('/', sep)+'.jpg', join(target, file).replace('/', sep)+'.jpg')

copy_to_tagret(IMG_ROOT, TGT_ROOT_TRAIN, train_files)
copy_to_tagret(IMG_ROOT, TGT_ROOT_TEST, test_files)