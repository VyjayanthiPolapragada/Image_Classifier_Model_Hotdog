#main

#importing created python files

from display_image import *
from dataset_hotdog import *
from  image_classifier_model import *

#importing libraries
from torchvision import transforms
from torch.utils.data import random_split
from torch import manual_seed
from torchvision import models
from torch.nn import Linear
from torch.nn import CrossEntropyLoss
from torch.optim import Adam



mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(degrees=5),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)
                               ])

dataset_full = HotDogDataset("hotdognothotdogfull", transform = composed)

manual_seed(0)
training_size = int(len(dataset_full) * 0.7)
validation_size = int(len(dataset_full) * 0.15)
test_size = len(dataset_full) - training_size - validation_size
training_set, validation_set, test_set = random_split(dataset=dataset_full, lengths=(training_size, validation_size, test_size))

imshow(training_set[0][0])

# Batch size: train set  
batch_size = 50

# Learning rate  
lr = 5e-3

# Number epochs 
n_epochs = 25

model = models.resnet18(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

# Next, we set `n_classes` to the number of classes we have. 
# Recall that we have two classes: "hotdog" and "not hotdog".
n_classes = dataset_full.n_classes

# Now that we have those parameters set, we can replace the output layer, 
# `model.fc` of the neural network, with a `nn.Linear` object to classify 
# `n_classes`'s different classes. For the first parameter, known as **in_features**, 
# we input 512 because the second last hidden layer of the neural network 
# has 512 neurons.
model.fc = Linear(512, n_classes)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = lr)

trainer = ClassificationModelTrainer(model,
                                     training_set,
                                     validation_set,
                                     batch_size,
                                     criterion,
                                     optimizer)


#training the model

trainer.load_from("twenty-five-iters.pt")
trainer.train_model(n_epochs = 2)

#plot the training result

trainer.plot_training_stat()

#testing the model

accuracy = trainer.test(test_set)

#final result

print(f"The model reached an accuracy rate of {accuracy:.2f}% on images it has never seen before.")












