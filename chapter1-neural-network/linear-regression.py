import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Training data
age = np.array([ 5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
height = np.array([125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0]) + np.random.normal(0, 2, size=len(age))

age_train=torch.from_numpy(age).float()
height_train=torch.from_numpy(height).float()

# Build Linear Regression Model
class LinearRegression(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(1, 1)
  def forward(self,x):
    out=self.linear(x)
    return out

model=LinearRegression()

# Loss Function and Optimizer
loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.005)

# Training loop
num_epochs=10000
for epoch in range(num_epochs):
  model.train()
  out=model(age_train.unsqueeze(1)) # Reshape age_train
  loss=loss_fn(out,height_train.unsqueeze(1)) # Reshape height_train and use loss_fn
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if (epoch+1)%1000 ==0:
    print(f'Epoch: {epoch}, Loss: {loss.item():.6f}')

# Print final model parameters
model.eval()
print("\nFinal Model Parameters:")
print(model.state_dict())

# Model Evaluation
model.eval()
with torch.no_grad():
    predicted_height = model(age_train.unsqueeze(1)).squeeze(1)

fig = plt.figure(figsize=(10, 5))
plt.scatter(age, height, color='red', label='Training Data')
plt.plot(age, predicted_height.numpy(), color='blue', label='Regression Line')

plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Height (cm)', fontsize=12)
plt.title('Linear Regression Fit', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Predict new height
new_age=19
new_age_tensor=torch.tensor([new_age],dtype=torch.float32).unsqueeze(1)
model.eval()
with torch.no_grad():
  predicted_height_new=model(new_age_tensor)
print(f"Predicted height for age {new_age}: {predicted_height_new.item():.2f} cm")
