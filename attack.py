from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method

import tensorflow as tf
import numpy as np

def pgd(model, img, eps):
    
    img = tf.expand_dims(img, 0)

    pgd_data = projected_gradient_descent(model, img, eps, 0.01, 40, np.inf)

    return pgd_data[0]

def fgsm(model, img, eps):
    """
    untargeted FGSM의 적대적 예제 생성 함수

    :model: 학습된 인공지능 모델.
            공격자는 인공지능 모델의 모든 파라미터 값을 알고있음.
    :img:   적대적 예제로 바꾸고자 하는 이미지 데이터
    :eps:   적대적 예제에 포함될 noise 크기 결정.
            eps가 크면 클 수록, 적대적 공격은 성공률이 높지만,
            적대적 예제의 시각적 표현이 높아지는 단점이 있음.
    :return: tensor 형태의 적대적 예제

    """

    img = tf.expand_dims(img, 0)

    fgsm_data = fast_gradient_method(model, img, eps, np.inf)

    return fgsm_data[0]


def cw(model, img):

    img = tf.expand_dims(img, 0)
    img = tf.cast(img, tf.float32)

    cw_data = carlini_wagner_l2(model, img)

    return cw_data[0]
def mim(model, img, eps):
    
    img = tf.expand_dims(img, 0)
    img = tf.cast(img, tf.float32)

    mim_data = momentum_iterative_method(model, img, eps)

    return mim_data[0]

def bim(model, img, eps):
    
    img = tf.expand_dims(img, 0)

    bim_data = basic_iterative_method(model, img, eps)

    return bim_data[0]
