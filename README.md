# Testing-out-CNN

Requirements:
pip install torch torchvision


Here are the results:

![image](https://github.com/Reyzenello/Testing-out-CNN/assets/43668563/4560beb2-fc55-441f-bb6a-f5058899e8a2)


This code trains a Convolutional Neural Network (CNN) on the MNIST dataset using PyTorch. Here's a step-by-step explanation:

**1. Importing Libraries:** Imports necessary libraries from PyTorch for neural networks, optimization, data loading, and transformations.

**2. Hyperparameters:** Sets key parameters like `batch_size`, `learning_rate`, and `num_epochs`.

**3. Data Loading and Transformation:**

```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# ...
```
* `transforms.Compose`: Chains together transformations to apply to the data.
* `transforms.ToTensor()`: Converts the PIL images from the dataset into PyTorch tensors (numerical representation suitable for computation).
* `transforms.Normalize((0.5,), (0.5,))`: Normalizes the pixel values to have a mean of 0.5 and a standard deviation of 0.5. This is a common preprocessing step that often helps with training.

```python
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# ...
```

* `torchvision.datasets.MNIST`: Loads the MNIST dataset.
    * `root='./data'`: Specifies where to store the downloaded data.
    * `train=True`: Loads the training set.  `train=False` loads the test set.
    * `transform=transform`: Applies the defined transformations.
    * `download=True`: Downloads the dataset if it's not already present.

```python
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# ...
```
* `DataLoader`: Creates a data loader that handles batching and shuffling the training and test datasets.


**4. Defining the CNN Model (`CNN` class):**

```python
class CNN(nn.Module):
    # ...
```
* `nn.Module`:  Base class for all neural network modules in PyTorch.

```python
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # ...
```
* `__init__`: Constructor where layers are defined.
* `nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)`: A 2D convolutional layer.
    * `1`: Input channels (grayscale images).
    * `32`: Output channels (number of filters).
    * `kernel_size=3`: Size of the convolutional kernel (3x3).
    * `stride=1`: Stride of the convolution.
    * `padding=1`: Adds padding to preserve input dimensions.
* Similar logic for the `conv2` layer with 64 output channels.
* `self.pool = nn.MaxPool2d(...)`: A 2D max-pooling layer to downsample the feature maps.
* `self.fc1 = nn.Linear(...)`, `self.fc2 = nn.Linear(...)`: Fully connected (linear) layers.

```python
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # ...
```

* `forward`: Defines the forward pass (how data flows through the network).
* `F.relu`: Applies the ReLU activation function.
* `x.view(-1, 64 * 7 * 7)`: Flattens the 3D tensor output from the convolutional layers into a 1D tensor to be fed into the fully connected layers. The -1 is for auto calculating the batch size so that the number of elements will be the same.


**5. Initialization:**

```python
model = CNN()
criterion = nn.CrossEntropyLoss()  # Loss function (for multi-class classification)
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer with specified learning rate
```

**6. Training Loop:**

```python
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # ...
```
Iterates over epochs and batches of data.

* `outputs = model(data)`: Forward pass (get model predictions).
* `loss = criterion(outputs, targets)`: Calculate the loss.
* `optimizer.zero_grad()`: Clear gradients from the previous step.
* `loss.backward()`: Backpropagation (calculate gradients).
* `optimizer.step()`: Update model parameters based on the gradients.

**7. Evaluation:**

```python
model.eval() #set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation during evaluation. It is useful for inference because it reduces memory consumption.
    # ...
```
Evaluates the trained model on the test set.

**8. Saving the Model:**

```python
torch.save(model.state_dict(), 'cnn_model.pth')  #Saves the model's parameters to a file
```
