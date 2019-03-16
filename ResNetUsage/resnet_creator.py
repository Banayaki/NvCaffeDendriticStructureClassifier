from caffe.proto import caffe_pb2
import resnet

def save_proto(proto, prototxt):
    with open(prototxt, 'w') as f:
        f.write(str(proto))


if __name__ == '__main__':
    
    model = resnet.ResNet('resources/train.txt', 'resources/test.txt', 50)

    train_proto = model.resnet_layers_proto(8)

    test_proto = model.resnet_layers_proto(8, phase='TEST')

    save_proto(train_proto, 'proto/train.prototxt')

    save_proto(test_proto, 'proto/test.prototxt')