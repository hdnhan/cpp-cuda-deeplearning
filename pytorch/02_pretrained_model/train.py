import torch
import torchvision
import time


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.premodel = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.premodel(x)


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
])

trainset = torchvision.datasets.MNIST(
    root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(
    root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

epochs = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = Net().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


for e in range(epochs):
    start = time.time()

    train_loss, train_acc = 0, 0
    test_loss, test_acc = 0, 0

    net.train()
    for images, labels in trainloader:
        images = images.to(device=device)
        labels = labels.to(device=device, dtype=torch.long)

        output = net(images)
        loss = loss_fn(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (labels == output.argmax(dim=1)).sum().item()

    train_loss /= len(trainloader)
    train_acc /= len(trainloader.dataset)

    # for python deployment
    torch.save(net.state_dict(), './checkpoint/model.pt')
    # for cpp deployment
    image = torch.rand(1, 1, 28, 28).to(device=device)
    # image, _ = next(iter(trainloader))
    traced_net = torch.jit.trace(net, image.to(device))
    traced_net.save('./checkpoint/traced_model.pt')

    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device=device)
            labels = labels.to(device=device, dtype=torch.long)

            output = net(images)
            loss = loss_fn(output, labels)

            test_loss += loss.item()
            test_acc += (labels == output.argmax(dim=1)).sum().item()
    test_loss /= len(testloader)
    test_acc /= len(testloader.dataset)

    end = time.time()
    print('Epoch {}/{}: train_loss: {:.4f} train_acc: {:.4f} test_loss: {:.4f} test_acc: {:.4f} time {:.4f}'
          .format(e + 1, epochs, train_loss, train_acc, test_loss, test_acc, end - start))
