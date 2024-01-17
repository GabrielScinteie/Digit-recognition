import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import CNN

# transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

# download and load the training data
trainset = datasets.MNIST('data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# download and load the test data
testset = datasets.MNIST('data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
print(len(testset))
print(len(trainset))
# show an image from the MNIST dataset
for i in range(100, 110):
    plt.imshow(trainset[i][0].numpy().squeeze(), cmap='gray')
    plt.show()
#
# # Create an instance of the CNN
# model = CNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Training Model.
# epochs = 10
# losses = []
# for e in range(epochs):
#     running_loss = 0
#     for images, labels in trainloader:
#         optimizer.zero_grad()  # Zeros out the gradient
#
#         output = model(images)  # Pass image through the model
#         loss = criterion(output, labels)  # Loss function calculation
#         loss.backward()  # Derivative of loss using backpropagation
#         optimizer.step()  # Updates model parameters, using gradients from backward()
#
#         running_loss += loss.item()
#     else:
#         average_loss = running_loss / len(trainloader)
#         print(f"Training loss: {average_loss}")
#         losses.append(average_loss)
#
# # Visualization
# plt.figure(figsize=(10, 5))
# plt.plot(range(epochs), losses)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss over Time')
# plt.show()
#
# correct_count, all_count = 0, 0
# for images, labels in testloader:
#     for i in range(len(labels)):
#         img = images[i].view(1, 1, 28, 28)  # Reshape for grayscale image
#         with torch.no_grad():
#             logps = model(img)  # Make the prediction
#         ps = torch.exp(logps)  # Calculate actual probabilities
#         probab = list(ps.numpy()[0])  # Tensor -> List()
#         pred_label = probab.index(max(probab))  # Find the highest prediction label
#         true_label = labels.numpy()[i]  # Get the true label.
#         if (true_label == pred_label):  # Check if the prediction is correct.
#             correct_count += 1
#         all_count += 1
#
#
# print("Number of Images Tests =", all_count)
# print("\nModel Accuracy =", (correct_count / all_count))
#
# torch.save(model.state_dict(), 'trained_model.pth')  # Saving model