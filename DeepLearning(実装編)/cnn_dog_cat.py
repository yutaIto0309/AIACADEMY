#%%
from icrawler.builtin import BingImageCrawler

# 猫の画像100枚を取得
crawler = BingImageCrawler(storage={'root_dir':'cat'})
crawler.crawl(keyword='猫', max_num=100)
# %%
# 犬の画像を100枚を取得
crawler = BingImageCrawler(storage={'root_dir':'dog'})
crawler.crawl(keyword='犬', max_num=100)
# %%
# 画像の表示
import os
from IPython.display import Image,display_jpeg
path = os.getcwd()
display_jpeg(Image(path + '\\cat\\000001.jpg'))
# %%
from PIL import Image, ImageFile
import glob
import numpy as np 

ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ['dog', 'cat']
num_classes = len(classes)
image_size = 64
num_testdata = 25

X_train, X_test, y_train, y_test = [],[],[],[]

for index, classlabel in enumerate(classes):
    photo_dir = path + '\\' + classlabel
    files = glob.glob(photo_dir + '\\*.jpg')
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert('RGB')
        image = image.resize((image_size, image_size))
        data  = np.asarray(image)
        if i < num_testdata:
            X_test.append(data)
            y_test.append(index)
        else:
            # 画像の角度変更や反転によってテストパターンを増やす
            for angle in range(-20, 20, 5):
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                y_train.append(index)
                image_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data =  np.asarray(image_trains)
                X_train.append(data)
                y_train.append(index)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

xy = (X_train, X_test, y_train, y_test)
np.save('dog_cat.npy', xy)
# %%
