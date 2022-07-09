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

testset = torchvision.datasets.MNIST(
    root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = torch.jit.load('./checkpoint/model.pt').to(device)
loss_fn = torch.nn.CrossEntropyLoss()

net.eval()
test_loss, test_acc = 0, 0

start = time.time()
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
print('test_loss: {:.4f} test_acc: {:.4f} time: {:.4f}'.format(
    test_loss, test_acc, end-start))
