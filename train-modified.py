# %%
# Import necessary packages
import matplotlib
matplotlib.use("Agg")
# %%
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50, InceptionV3
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
import cv2
import os
# %%
# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
# %%
# Setup paths based on the new structure
BASE_PATH = 'CCDEPLRL_PROJECT'
args = {
	"dataset": os.path.join(BASE_PATH, "dataset", "frames"),
	"model": os.path.join(BASE_PATH, "model"),
    "label-bin": os.path.join(BASE_PATH, "model", "lb.pickle"),
    "epochs": 25,
    "plot": os.path.join(BASE_PATH, "model", "training_plot.png")
}
# %%
# Ensure the model directory exists
os.makedirs(args["model"], exist_ok=True)
# %%
# Initialize the set of labels
LABELS = set(["Violence", "NonViolence"])
# %%
# Load images
print('-'*100)
print("[INFO] loading images...")
print('-'*100)
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
# %%
# Loop over the image paths
for imagePath in tqdm(imagePaths):
	# Extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# If the label is not part of our labels of interest, ignore
	if label not in LABELS:
		continue

	# Load the image, convert to RGB, and resize to 224x224
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# Update the data and labels lists
	data.append(image)
	labels.append(label)
# %%
# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)
# %%
# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# %%
# Split the data into training and testing sets (75% train, 25% test)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)
# %%
# Initialize training data augmentation
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
# %%
# Initialize validation/testing data augmentation
valAug = ImageDataGenerator()
# %%
# Define the ImageNet mean subtraction (RGB order)
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean
# %%
# Load the InceptionV3 network without the top FC layer
baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# %%
# Add custom head to the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
# %%
# Create the actual model
model = Model(inputs=baseModel.input, outputs=headModel)
# %%
# Make whole model trainable
model.trainable = True
# %%
# Compile the model
print('-'*100)
print("[INFO] compiling model...")
print('-'*100)
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print(model.summary())
# %%
# Train the model
print('-'*100)
print("[INFO] training model...")
print('-'*100)
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=args["epochs"])
# %%
# Evaluate the network
print('-'*100)
print("[INFO] evaluating network...")
print('-'*100)
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
# %%
# Plot the training loss and accuracy
print('-'*100)
print("[INFO] plotting training loss and accuracy...")
print('-'*100)
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
# %%
# Save the model
print('-'*100)
print("[INFO] saving model...")
print('-'*100)
model.save(os.path.join(args["model"], "violence_detection.h5"))
# %%
# Save the label binarizer
f = open(args["label-bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
# %%
print("Training completed successfully!")
