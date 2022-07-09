import torch
import torchvision
import time


transform = torchvision.transforms.Compose([
    torchvision.transforms.Pad(4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32),
    torchvision.transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = torchvision.models.resnet18(num_classes=10).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=8, gamma=1.0/3)

epochs = 10
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
    torch.save(net.state_dict(), './checkpoint/model_py.pt')
    scheduler.step()

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