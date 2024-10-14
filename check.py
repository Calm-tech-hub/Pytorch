from torchvision import transforms
trans_tensor = transforms.ToTensor()
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
trans_resize = transforms.Resize((512,512))
trans_random = transforms.RandomCrop(512)
