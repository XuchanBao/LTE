from spaghettini import quick_register

from torchvision import transforms


@quick_register
def resize_normalize():
    return transforms.Compose([transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,),
                                                    (0.5,))])
