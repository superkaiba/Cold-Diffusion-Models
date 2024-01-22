import torchvision
import os
import errno
import shutil
from pathlib import Path
from PIL import Image

DATA_FOLDER = '/home/mila/t/thomas.jiralerspong/scratch/ood_diffusion/cold_diffusion/data/'
# DATA_FOLDER="./data"
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass


CelebA_folder = '/fs/cml-datasets/CelebA-HQ/images-128/' # change this to folder which has CelebA data

############################################# MNIST ###############################################
trainset = torchvision.datasets.MNIST(
            root=DATA_FOLDER, train=True, download=True)
root = F'{DATA_FOLDER}/root_mnist/'
del_folder(root)
create_folder(root)

for i in range(10):
    lable_root = root + str(i) + '/'
    create_folder(lable_root)

for idx in range(len(trainset)):
    img, label = trainset[idx]
    print(idx)
    img.save(root + str(label) + '/' + str(idx) + '.png')


trainset = torchvision.datasets.MNIST(
            root=DATA_FOLDER, train=False, download=True)
root = f'{DATA_FOLDER}/mnist_test/'
del_folder(root)
create_folder(root)

for i in range(10):
    lable_root = root + str(i) + '/'
    create_folder(lable_root)

for idx in range(len(trainset)):
    img, label = trainset[idx]
    print(idx)
    img.save(root + str(label) + '/' + str(idx) + '.png')


############################################# Cifar10 ###############################################
trainset = torchvision.datasets.CIFAR10(
            root=DATA_FOLDER, train=True, download=True)
root = f'{DATA_FOLDER}/root_cifar10/'
del_folder(root)
create_folder(root)

for i in range(10):
    lable_root = root + str(i) + '/'
    create_folder(lable_root)

for idx in range(len(trainset)):
    img, label = trainset[idx]
    print(idx)
    img.save(root + str(label) + '/' + str(idx) + '.png')


trainset = torchvision.datasets.CIFAR10(
            root=DATA_FOLDER, train=False, download=True)
root = f'{DATA_FOLDER}/root_cifar10_test/'
del_folder(root)
create_folder(root)

for i in range(10):
    lable_root = root + str(i) + '/'
    create_folder(lable_root)

for idx in range(len(trainset)):
    img, label = trainset[idx]
    print(idx)
    img.save(root + str(label) + '/' + str(idx) + '.png')


############################################# CelebA ###############################################
# root_train = f'{DATA_FOLDER}/root_celebA_128_train_new/'
# root_test = f'{DATA_FOLDER}/root_celebA_128_test_new/'
# del_folder(root_train)
# create_folder(root_train)

# del_folder(root_test)
# create_folder(root_test)

# exts = ['jpg', 'jpeg', 'png']
# folder = CelebA_folder
# paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

# for idx in range(len(paths)):
#     img = Image.open(paths[idx])
#     print(idx)
#     if idx < 0.9*len(paths):
#         img.save(root_train + str(idx) + '.png')
#     else:
#         img.save(root_test + str(idx) + '.png')
