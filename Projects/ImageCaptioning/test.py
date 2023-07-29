import torch
from train import FlamingoModel, FlamingoProcessor
from PIL import Image


print('preparing model...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = FlamingoModel.from_pretrained('./weights/small/flamingo-coco_2/checkpoint-618')
model = FlamingoModel.from_pretrained('./weights/large')

# print (model)
model.to(device)
model.eval()
processor = FlamingoProcessor(model.config)

# load and process an example image
print('loading image and generating caption...')


# image = Image.open(requests.get(url, stream=True).raw)


for i in range(9):
    
    image = Image.open('./Test_images/'+str(i+1)+'.jpg')
    caption = model.generate_captions(processor, images=[image], device=device)

    image.show()
    print('generated caption:', caption)
