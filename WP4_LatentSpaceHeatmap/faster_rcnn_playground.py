import torch
import torchvision


def main():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # For training
    images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
    labels = torch.randint(1, 91, (4, 11))
    images = list(image for image in images)
    targets = []
    for i in range(len(images)):
        d = {'boxes': boxes[i], 'labels': labels[i]}
        targets.append(d)
    output = model(images, targets)
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)


if __name__ == '__main__':
    main()
