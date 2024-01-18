import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import CNN

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

trainset = datasets.MNIST('data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
print(len(testset))
print(len(trainset))

# for i in range(100, 110):
#     plt.imshow(trainset[i][0].numpy().squeeze(), cmap='gray')
#     plt.show()

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
losses = []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()

        output = model(images) # Facem predictia
        loss = criterion(output, labels) # Calculam functie cost
        loss.backward() # Calculam gradienti
        optimizer.step() # Facem backpropagation

        running_loss += loss.item()
    else:
        average_loss = running_loss / len(trainloader)
        print(f"Training loss: {average_loss}")
        losses.append(average_loss)

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()

correct_count, all_count = 0, 0
for images, labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 1, 28, 28)
        with torch.no_grad():
            logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number of Images Tests =", all_count)
print("\nModel Accuracy =", (correct_count / all_count))

torch.save(model.state_dict(), 'trained_model.pth')