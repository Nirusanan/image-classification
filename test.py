from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np

model = load_model('traffic_sign.h5')

img_url = 'test_images/t1.png'

img = load_img(img_url, target_size=(30, 30, 3))

img = img_to_array(img) / 255

img = np.expand_dims(img, axis=0)

result = model.predict(img)


image_genrator = ImageDataGenerator(rescale=1 / 255)
image_data = image_genrator.flow_from_directory('./data', target_size=(30, 300, 3))
class_names = image_data.class_indices.items()
class_names = np.array([key.title() for key, value in class_names])

ans = np.argmax(result)

print(class_names[ans])

