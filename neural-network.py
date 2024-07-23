import torch 
from math import sqrt

class Neuron:
    def __init__(self, num_inputs, num_outputs):
        # Glorot init works better than He init, why??
        self.weights = torch.randn(num_inputs, num_outputs) / sqrt(num_inputs)
        self.bias = torch.zeros(num_outputs)

    def forward(self, inputs):
        self.input = inputs
        output = inputs @ self.weights + self.bias
        return output
    
    def backward(self, grad_output):
        self.input.g = grad_output @ self.weights.t()
        self.weights.g = self.input.t() @ grad_output
        self.bias.g = grad_output.sum(0)
        return self.input.g
    
class ReLU:
    def __call__(self, input):
        self.input = input
        self.output = input.clamp_min(0)
        return self.output
    
    def backward(self, grad_output):
        self.input.g = (self.input > 0).float() * grad_output
        return self.input.g

class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.neuron = Neuron(num_inputs, num_outputs)
        self.relu = ReLU()

    def __call__(self, inputs):
        linear_output = self.neuron.forward(inputs)
        return self.relu(linear_output)

    def backward(self, grad_output):
        relu_grad = self.relu.backward(grad_output)
        return self.neuron.backward(relu_grad)

    def update_params(self, lr):
        self.neuron.weights -= lr * self.neuron.weights.g
        self.neuron.bias -= lr * self.neuron.bias.g

class Model:
    def __init__(self, layer_sizes):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def update_params(self, lr):
        for layer in self.layers:
            layer.update_params(lr)

def mse_loss(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def mse_grad(predictions, targets):
    return 2 * (predictions - targets)/ targets.numel()

def create_dataset(num_samples=1000):
    x = torch.randn(num_samples, 10)
    y = (x.sum(dim=1) > 0).float().unsqueeze(1)
    return x, y

def train(model, x, y, lr=0.01, epochs=100, batch_size=32):
    # Shuffle data to introduce randomness
    for epoch in range(epochs):
        indices = torch.randperm(x.size(0))
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        for i in range(0, x.size(0), batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            predictions = model(x_batch)
            loss = mse_loss(predictions, y_batch)
            loss_grad = mse_grad(predictions, y_batch)
            
            model.backward(loss_grad)
            model.update_params(lr)

        if epoch % 10 == 0:
            with torch.no_grad():
                total_loss = mse_loss(model(x), y)
                print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

X, y = create_dataset()
model = Model([10, 5, 1])
train(model, X, y)