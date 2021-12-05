
import torch as T

class Model(T.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = T.nn.Sequential(*[T.nn.Conv2d(3, 8, 3, padding=1), 
                                       T.nn.BatchNorm2d(8), T.nn.ReLU(), 
                                       T.nn.Conv2d(8, 16, 3, padding=1), 
                                       T.nn.BatchNorm2d(16), T.nn.ReLU(), 
                                       T.nn.MaxPool2d(2)])
        self.conv2 = T.nn.Sequential(*[T.nn.Conv2d(16, 32, 3, padding=1), 
                                       T.nn.BatchNorm2d(32), T.nn.ReLU(), 
                                       T.nn.Conv2d(32, 64, 3, padding=1), 
                                       T.nn.BatchNorm2d(64), T.nn.ReLU(), 
                                       T.nn.MaxPool2d(2)])
        self.conv3 = T.nn.Sequential(*[T.nn.Conv2d(64, 128, 3, padding=1), 
                                       T.nn.BatchNorm2d(128), T.nn.ReLU(), 
                                       T.nn.Conv2d(128, 256, 3, padding=1), 
                                       T.nn.BatchNorm2d(256), T.nn.ReLU(), 
                                       T.nn.MaxPool2d(2)])
        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.25), 
                                    T.nn.Linear(4096, 1024), 
                                    T.nn.ReLU(), T.nn.Dropout(0.25), 
                                    T.nn.Linear(1024, 256), 
                                    T.nn.ReLU(), T.nn.Dropout(0.25), 
                                    T.nn.Linear(256, 10)])
    
    def forward(self, img):
        _B = img.shape[0]
        
        out = self.conv1(img)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out.view([_B, -1]))
        
        return out
    