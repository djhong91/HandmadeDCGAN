from model import DCGAN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "faceimage", "학습 데이터 파일 경로")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "체크포인트 관리 폴더의 경로")
flags.DEFINE_string("sample_dir", "samples", "체크포인트 관리 폴더의 경로")

flags.DEFINE_string("input_fname_pattern", "*.jpg", "입력 이미지 확장자")
flags.DEFINE_integer("input_height", 64, "이미지 세로 길이")
flags.DEFINE_integer("input_width", 64, "이미지 가로 길이")
flags.DEFINE_integer("output_height", 64, "출력 이미지 세로 길이")
flags.DEFINE_integer("output_width", 64, "출력 이미지 가로 길이")

flags.DEFINE_boolean("is_train", True, "학습 페이즈일 때 True, 테스트는 False")
flags.DEFINE_integer("batch_size", 100, "학습 배치 사이즈")

flags.DEFINE_integer("epoch", 100, "학습 epoch 수")
flags.DEFINE_integer("save_freq", 5, "checkpoint 저장 빈도. 단위는 epoch")
flags.DEFINE_boolean("continue_train", True, "학습을 이어서 진행할 때")

flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")

flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")

args = flags.FLAGS

def main():
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
            sess,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            input_fname_pattern=args.input_fname_pattern,
            batch_size=args.batch_size,
            dataset_dir=args.dataset_dir,
            checkpoint_dir=args.checkpoint_dir,
            sample_dir=args.sample_dir)

        if args.is_train:
            dcgan.train(args)
        else:
            dcgan.test(args)

if __name__ == '__main__':
    main()