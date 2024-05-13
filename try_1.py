from torchvision import models

print(dir(models))

resnet = models.resnet101(pretrained = True)

