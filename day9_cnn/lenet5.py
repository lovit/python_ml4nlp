class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.c2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.c3 = nn.Conv2d(16, 120, kernel_size=(4, 4))
        self.f4 = nn.Linear(120, 84)
        self.f5 = nn.Linear(84, 10)
        self.out = nn.LogSoftmax(dim=-1)

    def forward(self, img, debug=False):
        out = F.relu(self.c1(img))
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = F.relu(self.c2(out))        
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = F.relu(self.c3(out))
        out = out.view(img.size()[0], -1)

        out = F.relu(self.f4(out))
        out = self.f5(out)
        out = self.out(out)

        return out
