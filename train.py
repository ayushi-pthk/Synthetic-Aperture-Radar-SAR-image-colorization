import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize encoder and decoder
encoder = EnsembleEncoder().to(device)
decoder = Decoder().to(device)

# Freeze the encoder parameters as they are pre-trained
for param in encoder.parameters():
    param.requires_grad = False

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(decoder.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Training loop
num_epochs = 30
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training phase
    encoder.eval()
    decoder.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")
    for i, (L_batch, ab_batch) in enumerate(train_bar):
        L, ab = L_batch.to(device), ab_batch.to(device)
        L = L.repeat(1, 3, 1, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        features_56x56, features_28x28, features_14x14, features_7x7 = encoder(L)
        output = decoder(features_7x7, features_14x14, features_28x28, features_56x56)

        # Compute loss
        loss = criterion(output, ab)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate running loss
        running_loss += loss.item()

        # Update progress bar
        train_bar.set_postfix(loss=f"{running_loss/(i+1):.4f}")

    # Validation phase
    decoder.eval()
    val_loss = 0.0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")
    with torch.no_grad():
        for i, (L_batch, ab_batch) in enumerate(val_bar):
            L, ab = L_batch.to(device), ab_batch.to(device)
            L = L.repeat(1, 3, 1, 1)  # Replicate grayscale to 3 channels

            # Forward pass
            features_56x56, features_28x28, features_14x14, features_7x7 = encoder(L)
            output = decoder(features_7x7, features_14x14, features_28x28, features_56x56)

            # Compute validation loss
            loss = criterion(output, ab)
            val_loss += loss.item()

            val_bar.set_postfix(loss=f"{val_loss/(i+1):.4f}")

    # Calculate average losses
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(decoder.state_dict(), 'model_1.pth')
        print(f"Model saved with validation loss: {best_val_loss:.4f}")

    # Step the scheduler
    scheduler.step(avg_val_loss)

print("Training complete.")
