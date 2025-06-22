import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Hyperparameters
epochs = 50
batch_size = 128
lr = 0.0002
z_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Generator ====================
class Generator(nn.Module):
    def __init__(self, z_dim=100, label_dim=10):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(z_dim + label_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([z, label_input], dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

# ==================== Discriminator ====================
class Discriminator(nn.Module):
    def __init__(self, label_dim=10):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(784 + label_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_input = self.label_emb(labels)
        x = torch.cat([img_flat, label_input], dim=1)
        validity = self.model(x)
        return validity

# ==================== Prepare Dataset ====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==================== Initialize Models ====================
generator = Generator(z_dim=z_dim).to(device)
discriminator = Discriminator().to(device)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# ==================== Training Loop ====================
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size_curr = imgs.size(0)
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # Real and fake labels
        real = torch.ones(batch_size_curr, 1).to(device)
        fake = torch.zeros(batch_size_curr, 1).to(device)

        # === Train Generator ===
        z = torch.randn(batch_size_curr, z_dim).to(device)
        gen_labels = torch.randint(0, 10, (batch_size_curr,)).to(device)
        gen_imgs = generator(z, gen_labels)
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = criterion(validity, real)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # === Train Discriminator ===
        real_pred = discriminator(real_imgs, labels)
        d_real_loss = criterion(real_pred, real)

        fake_pred = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = criterion(fake_pred, fake)

        d_loss = (d_real_loss + d_fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Print log
        if i % 100 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

# ==================== Save Trained Generator ====================
os.makedirs("models", exist_ok=True)
torch.save(generator.state_dict(), "generator.pth")
print("✅ Generator saved to 'generator.pth'")