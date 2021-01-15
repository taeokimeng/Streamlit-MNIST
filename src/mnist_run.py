# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import streamlit as st

print(tf.__version__)
save_image_1 = None
save_image_2 = None

def run_mnist(selected_optimizer='adam', selected_metric='accuracy', selected_epochs=5):
    # 10개 클래스에 대한 예측을 모두 그래프로 표현
    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary, aspect="auto")

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]), color=color)

    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        plt.xticks([])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    train_images, train_labels, test_images, test_labels, class_names = prepare_running()
    col1, col2 = st.beta_columns((1, 1))

    # container_1 = st.beta_container()
    # container_2 = st.beta_container()

    # show_data_and_labels(train_images, train_labels, class_names)

    # First column
    with col1:
        with st.beta_expander("Show the first data."):
            show_data(train_images)

    # 신경망 모델에 주입하기 전에, 값의 범위를 0~1 사이로 조정
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Second column
    with col2:
        with st.beta_expander("Show the labels."):
            show_data_labels(train_images, train_labels, class_names)

    # Start running
    if st.sidebar.button("Start classification"):
        # 신경망의 기본 구성 요소인 층(layer) 설정
        # 첫 번째 층인 tf.keras.layers.Flatten은
        # 2차원 배열(28 x 28 픽셀)의 이미지 포맷을 28 * 28 = 784 픽셀의
        # 1차원 배열로 변환
        # 두 개의 tf.keras.layers.Dense 층이 연속되어 연결됨
        # 이 층을 밀집 연결(densely-connected) 또는 완전 연결(fully-connected)
        # 층이라고 부름
        # 첫 번째 Dense 층은 128개의 노드(또는 뉴런)를 가짐
        # 두 번째 (마지막) 층은 10개의 노드의 소프트맥스(softmax) 층
        # 이 층은 10개의 확률을 반환하고 반환된 값의 전체 합은 1
        # 각 노드는 현재 이미지가 10개 클래스 중 하나에 속할 확률을 출력
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        # 손실 함수(Loss function): 훈련 하는 동안 모델의 오차를 측정.
        # 모델의 학습이 올바른 방향으로 향하도록 이 함수를 최소화해야 함.
        # 옵티마이저(Optimizer): 데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정.
        # 지표(Metrics): 훈련 단계와 테스트 단계를 모니터링하기 위해 사용.
        model.compile(optimizer=selected_optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=[selected_metric])

        # 모델이 훈련 데이터를 학습
        model.fit(train_images, train_labels, epochs=selected_epochs)  # 예제 epochs=5

        # 정확도 평가
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print('\n테스트 정확도:', test_acc)

        # 훈련된 모델을 사용하여 이미지에 대한 예측
        predictions = model.predict(test_images)

        # 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
        # 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
        # First column
        with col1:
            st.subheader("Predict the 15 images")
            num_rows = 5
            num_cols = 3
            num_images = num_rows * num_cols
            fig5 = plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))  # figsize=(2 * 2 * num_cols, 2 * num_rows)
            for i in range(num_images):
                plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
                plot_image(i, predictions, test_labels, test_images)
                plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
                plot_value_array(i, predictions, test_labels)
            fig5.tight_layout() # For layout
            st.write(fig5)
            # To draw previous image
            global save_image_1
            if save_image_1 is not None:
                st.subheader("The previous 15 images")
                st.write(save_image_1)
            save_image_1 = fig5

        # 테스트 세트에서 이미지 하나를 선택합니다
        img = test_images[0]
        # print(img.shape)

        # 이미지 하나만 사용할 때도 배치에 추가합니다
        img = (np.expand_dims(img, 0))
        # print(img.shape)

        # 이미지 예측
        predictions_single = model.predict(img)
        # print(predictions_single)

        # Second column
        with col2:
            st.subheader("The detailed result about the first image ")
            fig6 = plt.figure(figsize=(8, 6)) # 5, 3
            plot_value_array(0, predictions_single, test_labels)
            fig6.tight_layout() # For layout
            _ = plt.xticks(range(10), class_names, rotation=45)
            st.write(fig6)
            # To draw previous image
            global save_image_2
            if save_image_2 is not None:
                st.subheader("The previous result")
                st.write(save_image_2)
            save_image_2 = fig6

    # print(predictions[0])
    # print(np.argmax(predictions[0]))
    # print(test_labels[0])

    # 이미지 예측 출력
    # i = 0
    # fig3 = plt.figure(figsize=(6, 3))
    # plt.subplot(1, 2, 1)
    # plot_image(i, predictions, test_labels, test_images)
    # plt.subplot(1, 2, 2)
    # plot_value_array(i, predictions, test_labels)
    # st.write(fig3)
    #
    # i = 12
    # fig4 = plt.figure(figsize=(6, 3))
    # plt.subplot(1, 2, 1)
    # plot_image(i, predictions, test_labels, test_images)
    # plt.subplot(1, 2, 2)
    # plot_value_array(i, predictions, test_labels)
    # st.write(fig4)

@st.cache
def prepare_running():
    # Below code is for "failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    fashion_mnist = keras.datasets.fashion_mnist
    # 훈련 세트에는 60,000개의 이미지가 있고, 각 이미지는 28x28 픽셀로 표현
    # 테스트 세트에는 10,000개의 이미지가 있고, 각 이미지는 28x28 픽셀로 표현
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return train_images, train_labels, test_images, test_labels, class_names

def show_data(train_images):
    st.subheader("The first image in the training set")
    fig1 = plt.figure(figsize=(8, 8))
    # plt.title("The first image in the training set")
    plt.imshow(train_images[0], aspect="auto")
    plt.colorbar()
    plt.grid(False)
    # st.write(fig1)
    st.pyplot(fig1)

def show_data_labels(train_images, train_labels, class_names):
    st.subheader("The first 25 images with the class name")
    # 훈련 세트에서 처음 25개 이미지와 그 아래 클래스 이름 출력
    fig2 = plt.figure(figsize=(8, 8))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary, aspect="auto")
        plt.xlabel(class_names[train_labels[i]])
    # plt.suptitle("The first 25 images from the training set and the class name", fontsize=20)
    # st.write(fig2)
    st.pyplot(fig2)
