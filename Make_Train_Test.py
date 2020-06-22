import glob, os

# create file train.txt and text.txt
# path_data is folder contain images
# percentage test is a percentage of images to be used for the test set


# Directory where the data will reside, relative to 'darknet.exe'
path_data = 'data/JP_Coin_Banknote/'

# Percentage of images to be used for the test set
percentage_test = 10;

# Create and/or truncate train.txt and test.txt
file_train = open('data/train_JP_Coin_Banknote.txt', 'w')  
file_test = open('data/test_JP_Coin_Banknote.txt', 'w')

# Populate train.txt and test.txt
counter = 1  
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(path_data+"/*.jpg"):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    print(pathAndFilename)
    #print("hello")
    if counter == index_test:
        counter = 1
        file_test.write(pathAndFilename + "\n")
    else:
        file_train.write(pathAndFilename + "\n")
        counter = counter + 1