import numpy as np
import tensorflow as tf
import saliency.core as saliency

def model_fn(images, call_model_args, expected_keys=None):
    """
    함수_vanilla_saliency 데이터 처리 함수
    인공지능 모델의 output과 input 사이의 gradient 연산 처리
    
    """

    target_class_idx = call_model_args['class']
    model = call_model_args['model']
    images = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output = model(images)
            output = output[:,target_class_idx]
            gradients = np.array(tape.gradient(output, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv, output = model(images)
            gradients = np.array(tape.gradient(output, conv))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}

def saliency_map(model, img):
    """
    Saliency map을 이용하여 기여도 맵 추출 함수

    :model: 학습된 인공지능 모델
            인공지능 모델이 바뀔 때, 기여도 맵 또한 변경됨.
    :img:   기여도 맵을 추출하고 하는 이미지 데이터
    :return: 추출된 기여도 맵

    """
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    grad = saliency.GradientSaliency()
    attr = grad.GetMask(img, model_fn, args)
    # attr = saliency.VisualizeImageGrayscale(attr)

    return tf.reshape(attr, (*attr.shape, 1))

def ig(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    baseline = np.zeros(img.shape)
    ig = saliency.IntegratedGradients()
    attr = ig.GetMask(img, model_fn, args, x_steps=25, x_baseline=baseline, batch_size=20)
    # attr = saliency.VisualizeImageGrayscale(attr)

    return tf.reshape(attr, (*attr.shape, 1))

def smooth_saliency_map(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    smooth_grad = saliency.GradientSaliency()
    smooth_attr = smooth_grad.GetSmoothedMask(img, model_fn, args)
    smooth_attr = saliency.VisualizeImageGrayscale(smooth_attr)

    return tf.reshape(smooth_attr, (*smooth_attr.shape, 1))

def smooth_ig(model, img):

    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    baseline = np.zeros(img.shape)
    smooth_ig = saliency.IntegratedGradients()

    smooth_attr = smooth_ig.GetSmoothedMask(
        img, model_fn, args, x_steps=25, x_baseline=baseline, batch_size=20)

    smooth_attr = saliency.VisualizeImageGrayscale(smooth_attr)

    return tf.reshape(smooth_attr, (*smooth_attr.shape, 1))
