import requests
from PIL import Image
from io import BytesIO

from transformers import pipeline

images = [
'/media/michael/data/images/classification_faces/Screenshot_20241214_222538.png',
'/media/michael/data/images/classification_faces/Screenshot_20241214_222653.png',
'/media/michael/data/images/classification_faces/Screenshot_20241214_222731.png',
'/media/michael/data/images/classification_faces/Screenshot_20241214_222819.png',
'/media/michael/data/images/classification_faces/Screenshot_20241214_222933.png']

def process_image(im):
    age_image_classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")
    age_classification_result = age_image_classifier(im)

    print(age_classification_result)

    gender_image_classifier = pipeline("image-classification", model="rizvandwiki/gender-classification")
    gender_classification_result = gender_image_classifier(im)

    print(gender_classification_result)

    luke_jacob_nsfw_image_classifier = pipeline("image-classification", model="LukeJacob2023/nsfw-image-detector")
    luke_jacob_nsfw_classification_result = luke_jacob_nsfw_image_classifier(im)

    print(luke_jacob_nsfw_classification_result)

# load images from file
for image in images:
    im = Image.open(image)
    process_image(im)