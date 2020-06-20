import tensorflow as tf
from keras.applications.densenet import DenseNet121, DenseNet169
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda
from keras.models import Model
from keras.optimizers import SGD


def load_model(dataset='melanoma', num_class=7, model='inceptionv3', mode='train', model_weights=None):
    if dataset == 'melanoma':
        input_shape = (299, 299, 3)
        if model == 'inceptionv3':
            base_model = InceptionV3(
                weights='imagenet', input_shape=input_shape, include_top=False)
        elif model == 'vgg16':
            base_model = VGG16(weights='imagenet',
                               input_shape=input_shape, include_top=False)
        elif model == 'vgg19':
            base_model = VGG19(weights='imagenet',
                               input_shape=input_shape, include_top=False)
        elif model == 'resnet50':
            base_model = ResNet50(weights='imagenet',
                                  input_shape=input_shape, include_top=False)
        elif model == 'inceptionresnetv2':
            base_model = InceptionResNetV2(
                weights='imagenet', input_shape=input_shape, include_top=False)
        elif model == 'densenet121':
            base_model = DenseNet121(
                weights='imagenet', input_shape=input_shape, include_top=False)
        elif model == 'densenet169':
            base_model = DenseNet169(
                weights='imagenet', input_shape=input_shape, include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_class, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    else:
        if model == 'inceptionv3':
            base_model = InceptionV3(weights='imagenet', include_top=False)
        elif model == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False)
        elif model == 'vgg19':
            base_model = VGG19(weights='imagenet', include_top=False)
        elif model == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False)
        elif model == 'inceptionresnetv2':
            base_model = InceptionResNetV2(
                weights='imagenet', include_top=False)
        elif model == 'densenet121':
            base_model = DenseNet121(weights='imagenet', include_top=False)
        elif model == 'densenet169':
            base_model = DenseNet169(weights='imagenet', include_top=False)
        base_model.layers.pop(0)  # remove input layer
        newInput = Input(batch_shape=(None, 299, 299, 1))
        x = Lambda(lambda image: tf.image.grayscale_to_rgb(image))(newInput)
        tmp_out = base_model(x)
        tmpModel = Model(newInput, tmp_out)
        x = tmpModel.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_class, activation='softmax')(x)
        model = Model(tmpModel.input, predictions)

    if mode == 'train':
        sgd = SGD(decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        for layer in model.layers:
            layer.trainable = True
    else:
        pass

    return model
