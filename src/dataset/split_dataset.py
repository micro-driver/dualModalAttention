import random
import os
import shutil
from tqdm import tqdm
import glob
path='E:\PythonProject\dualModalAttention\src\dataset'
train_path=path+'/train_set'
val_path=path+'/val_set'
test_path=path+'/test_set'

# def move_dir(from_dir, to_dir):
#   for img in tqdm(os.listdir(from_dir)):
#     shutil.move(os.path.join(from_dir, img), os.path.join(to_dir, img))
#
# move_dir('E:\PythonProject\dualModalAttention\src\dataset\Celeb-real\images', train_path)


def split_train_test(fileDir,tarDir):
  if not os.path.exists(tarDir):
      os.makedirs(tarDir)
  pathDir = os.listdir(fileDir)    #取图片的原始路径
  filenumber=len(pathDir)
  rate=0.1    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
  picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
  sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
  print("=========开始移动图片============")
  for name in tqdm(sample):
    shutil.move(os.path.join(fileDir, name), os.path.join(tarDir, name))
  print("=========移动图片完成============")
split_train_test(train_path, test_path)
split_train_test(train_path, val_path)

