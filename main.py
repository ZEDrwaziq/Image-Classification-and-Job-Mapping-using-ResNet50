import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input

model = ResNet50(weights='imagenet')


def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)

    decoded_prediction = decode_predictions(prediction, top=1)[0][0]

 
    return decoded_prediction[1], decoded_prediction[2]  



def map_to_job_class(predicted_class):
    job_mapping = {
        'lab_coat': 'Doctor/Scientist',
        'stethoscope': 'Doctor',
        'backpack': 'Student/Teacher',
        'television': 'Entertainer',
        'laptop': 'Engineer/Programmer',
        'military_uniform': 'soldier',
        'book': 'Teacher/Student',
        'fire_engine': 'Fier_man',
    }

    return job_mapping.get(predicted_class, 'Unknown Profession')



image_path = input("enter image path : ")  # Replace with the path to your image


predicted_class, probability = classify_image(image_path)

job_class = map_to_job_class(predicted_class)

print(f'The predicted class is: {predicted_class} with a probability of {probability * 100}',"%")
print(f'The mapped job class is: {job_class}')
