import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=5):
        super(Encoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)  # Assuming input size is 32x32
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, x, size_mask, difficulty_map):
        # Combine input with size mask and difficulty map
        x = torch.cat([x * size_mask, size_mask, difficulty_map], dim=1)
        
        # Forward pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Output mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=5, output_channels=1, max_width=9, max_height=11):
        super(Decoder, self).__init__()
        
        # Fully connected layer to upsample latent vector
        self.fc = nn.Linear(latent_dim + max_width + max_height + 1, 64 * 8 * 8)
        
        # Transposed convolutional layers
        self.tconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv3 = nn.ConvTranspose2d(16, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z, width_one_hot, height_one_hot, difficulty):
        # Concatenate latent vector with condition information
        condition_info = torch.cat([z, width_one_hot, height_one_hot, difficulty], dim=1)
        
        # Upsample using fully connected layer
        x = F.relu(self.fc(condition_info))
        x = x.view(x.size(0), 64, 8, 8)  # Reshape to match transposed conv input
        
        # Forward pass through transposed convolutional layers
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = self.tconv3(x)  # No activation on the last layer
        
        return x

class AvalonGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, latent_dim=5, max_width=9, max_height=11):
        super(AvalonGenerator, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_channels=output_channels, max_width=max_width, max_height=max_height)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, size_mask, difficulty_map, width_one_hot, height_one_hot, difficulty):
        # Encode
        mu, logvar = self.encoder(x, size_mask, difficulty_map)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed_x = self.decoder(z, width_one_hot, height_one_hot, difficulty.unsqueeze(-1))
        
        return reconstructed_x, mu, logvar

# # Example usage
# if __name__ == "__main__":
#     # Hyperparameters
#     batch_size = 4
#     input_channels = 1
#     output_channels = 1
#     latent_dim = 5
#     max_width = 9
#     max_height = 11
    
#     # Dummy inputs
#     x = torch.rand(batch_size, input_channels, 32, 32)  # Input levels (e.g., 32x32 grid)
#     size_mask = torch.randint(0, 2, (batch_size, 1, 32, 32)).float()  # Size mask
#     difficulty_map = torch.rand(batch_size, 1, 32, 32)  # Difficulty map
#     width_one_hot = torch.zeros(batch_size, max_width).scatter_(1, torch.randint(0, max_width, (batch_size, 1)), 1)
#     height_one_hot = torch.zeros(batch_size, max_height).scatter_(1, torch.randint(0, max_height, (batch_size, 1)), 1)
#     difficulty = torch.rand(batch_size)  # Normalized difficulty value [0, 1]
    
#     # Initialize model
#     model = AvalonGenerator(input_channels=input_channels, output_channels=output_channels, latent_dim=latent_dim, max_width=max_width, max_height=max_height)
    
#     # Forward pass
#     reconstructed_x, mu, logvar = model(x, size_mask, difficulty_map, width_one_hot, height_one_hot, difficulty)
    
#     print("Reconstructed shape:", reconstructed_x.shape)
#     print("Mu shape:", mu.shape)
#     print("Logvar shape:", logvar.shape)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the Encoder and Decoder classes as before

class AvalonGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, latent_dim=5, max_width=9, max_height=11):
        super(AvalonGenerator, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_channels=output_channels, max_width=max_width, max_height=max_height)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, size_mask, difficulty_map, width_one_hot, height_one_hot, difficulty):
        # Encode
        mu, logvar = self.encoder(x, size_mask, difficulty_map)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed_x = self.decoder(z, width_one_hot, height_one_hot, difficulty.unsqueeze(-1))
        
        return reconstructed_x, mu, logvar

def compute_loss(reconstructed_x, x, mu, logvar):
    # Reconstruction loss (cross-entropy for categorical distribution)
    reconstruction_loss = F.cross_entropy(reconstructed_x.view(-1, K), x.view(-1))
    
    # KL divergence loss
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return reconstruction_loss + kl_divergence

def train(model, train_loader, epochs=24000, checkpoint_interval=500, learning_rate=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, size_mask, difficulty_map, width_one_hot, height_one_hot, difficulty) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed_x, mu, logvar = model(x, size_mask, difficulty_map, width_one_hot, height_one_hot, difficulty)
            
            # Compute loss
            loss = compute_loss(reconstructed_x, x, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')
        
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), f'model_checkpoint_{epoch+1}.pth')

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 100
    input_channels = 1
    output_channels = 1
    latent_dim = 5
    max_width = 9
    max_height = 11
    K = 10  # Number of categorical tile classes
    
    # Dummy inputs
    x = torch.randint(0, K, (batch_size, 1, 32, 32)).float()  # Input levels (e.g., 32x32 grid)
    size_mask = torch.randint(0, 2, (batch_size, 1, 32, 32)).float()  # Size mask
    difficulty_map = torch.rand(batch_size, 1, 32, 32)  # Difficulty map
    width_one_hot = torch.zeros(batch_size, max_width).scatter_(1, torch.randint(0, max_width, (batch_size, 1)), 1)
    height_one_hot = torch.zeros(batch_size, max_height).scatter_(1, torch.randint(0, max_height, (batch_size, 1)), 1)
    difficulty = torch.rand(batch_size)  # Normalized difficulty value [0, 1]
    
    # Initialize model
    model = AvalonGenerator(input_channels=input_channels, output_channels=output_channels, latent_dim=latent_dim, max_width=max_width, max_height=max_height)
    
    # Create a dataset and dataloader
    dataset = TensorDataset(x, size_mask, difficulty_map, width_one_hot, height_one_hot, difficulty)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    train(model, train_loader, epochs=24000, checkpoint_interval=500, learning_rate=1e-5)