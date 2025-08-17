import os
import cv2 as cv
from PIL import Image
from imagehash import dhash
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import numpy as np
from keras.preprocessing.image import img_to_array

#Saving pictures in a directory or file
save_dir = 'C:\\Users\\Majesty\\Desktop\\Practice\\py project\\adcam1\\'
os.chdir(save_dir)

cam = cv.VideoCapture(0)

cv.namedWindow('My Camera')
img_counter = 0

while True:
    ret, frame = cam.read()

    if not ret:
        print('No picture captured')
        break

    cv.imshow('test', frame)
    k = cv.waitKey(1)

    if k % 256 == 27:  # Press escape key
        print('Camera has been shut down')
        break

    elif k % 256 == 32:
        img_name = "PIC_{}.jpg".format(img_counter)
        cv.imwrite(img_name, frame)
        print('Image has been taken for process')
        img_counter += 1

cam.release()
cv.destroyAllWindows()
image_array = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)
image_array = img_to_array(image_array)
image_array = np.expand_dims(image_array, axis=0)
image_array = image_array / 255.0


def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(directory, filename)
            img = cv.imread(img_path)
            images.append(img)
    return images

# Function to compare two images using dhash
def compare_images(image1, image2):
    hash1 = dhash(Image.fromarray(cv.cvtColor(image1, cv.COLOR_BGR2RGB)))
    hash2 = dhash(Image.fromarray(cv.cvtColor(image2, cv.COLOR_BGR2RGB)))
    # Calculate the Hamming distance
    diff = hash1 - hash2
    # Normalize the difference to a percentage scale
    similarity = 1.0 - (diff / 64.0)  # 64-bit hash
    return similarity * 100  # Convert to percentage

# Function to preprocess an image
def preprocess_image(image):
    # Preprocess the image (e.g., resize, normalize) to match the input format expected by your model
    # Example: resize to 100x100 pixels and normalize pixel values to [0, 1]
    resized_image = cv.resize(image, (100, 100))
    normalized_image = resized_image.astype('float32') / 255.0
    return normalized_image

# Load the model
model_path = r'C:\Users\Majesty\Desktop\Practice\py project\keras_model.h5'
model = load_model(model_path)
#model = keras.models.load_model(model_path)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Predicts the model
prediction = model.predict(image_array)

# Path to the directories containing the images
dir1 = 'C:\\Users\\Majesty\\Desktop\\Practice\\py project\\adcam1\\'
dir100 ='C:\\Users\\Majesty\\Desktop\\Practice\\py project\\hundred\\'
dir200='C:\\Users\\Majesty\\Desktop\\Practice\\py project\\Twohundred\\'
dir500='C:\\Users\\Majesty\\Desktop\\Practice\\py project\\500Hun\\'
dir50='C:\\Users\\Majesty\\Desktop\\Practice\\py project\\50fifty\\'

x=int(input('Enter the directory u wanted to choose(50,100,200,500):'))
if x==100:
    Y=dir100
elif x==200:
    Y=dir200  
    
elif x==50:
    Y=dir50
    
elif x==500:
    Y=dir500

# Load images from directories
images1 = load_images_from_directory(dir1)
images2 = load_images_from_directory(Y)

threshold = 50  # Set your threshold for similarity

'''for img1 in images1:
    similarities = []
    for img2 in images2:
        similarity = compare_images(img1, img2)
        similarities.append(similarity)
    
    avg_similarity = sum(similarities) / len(similarities)
    #print(f"Avg Similarity for {img1}: {avg_similarity:.2f}%")
    
    if avg_similarity < threshold:
        print("Fake currency!")
    else:
        print("real currency")'''

for img1 in images1:
    for img2 in images2:
        similarity = compare_images(img1, img2)
        if similarity > threshold:  # Set your threshold for similarity
            print("Similarity:", similarity)
            # Preprocess the image
            processed_img1 = preprocess_image(img1)
            processed_img2 = preprocess_image(img2)
            # Reshape the image for the model input
            input_img1 = np.expand_dims(processed_img1, axis=0)
            input_img2 = np.expand_dims(processed_img2, axis=0)
            # Make predictions using the model
            prediction1 = model.predict(input_img1)
            prediction2 = model.predict(input_img2)
            # Handle the predictions to identify fake currency
            if (prediction1 == 1).all() and (prediction2 == 1).all(): 
                print("Fake currency detected!")
            else:
                print("Not fake currency")
        '''else:
            print(f"Similarity: {similarity:.2f}%")'''
            
