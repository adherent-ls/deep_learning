
def adv():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Model().model
    model.load_state_dict(torch.load('/home/data/data/cifar10/best-baseline.pth'))

    batch_size = 1
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='/home/data/data/cifar10', train=False,
                                           download=True, transform=transform)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   shuffle=False, num_workers=2)

    weight = model.fc.weight
    linear_pos = nn.Linear(model.fc.in_features, model.fc.out_features, bias=False)
    linear_neg = nn.Linear(model.fc.in_features, model.fc.out_features, bias=False)
    weight_pos = (weight > 0) * weight
    weight_neg = (weight < 0) * weight

    linear_pos.weight = nn.Parameter(weight_pos)
    linear_neg.weight = nn.Parameter(weight_neg)

    model.linear_pos = linear_pos
    model.linear_neg = linear_neg

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # since we're not training, we don't need to calculate the gradients for our outputs
    model.eval()

    adv_n = 0
    t = 0
    for data in valid_dataloader:
        image, label = data
        image = image.to(device)
        label = label.to(device)

        noise = 0
        pre_d = None
        pre_x2 = None
        for i in range(100):
            image_r = image + noise
            image_r.requires_grad = True
            x2, pred_t = model(image_r)
            pred_pos = linear_pos(x2)
            pred_neg = linear_neg(x2)

            pred = pred_pos.detach() + pred_neg + model.fc.bias
            loss = criterion(pred, pred.max(dim=-1)[1])
            loss.backward()

            if pre_d is not None and pre_d[2].max(dim=-1)[1] != pred.max(dim=-1)[1]:
                adv_n += 1
                break
            pre_d = [pred_neg, pred_pos, pred]
            if pre_x2 is None:
                pre_x2 = x2
            grad = image_r.grad
            noise += grad
            noise = torch.clip(noise, -0.1, 0.1)
        t += 1
        print(adv_n, t, adv_n / t)
