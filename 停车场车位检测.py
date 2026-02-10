import glob
import pickle

from keras.src.saving import load_model
from matplotlib import pyplot as plt

from Parking import Parking
def img_process(test_images, park):
    """
    完整的图像处理流程
    顺序：颜色选择→灰度化→边缘检测→ROI→霍夫变换→识别车位
    """
    # 1. 颜色选择
    white_yellow_images = []
    for img in test_images:
        result = park.select_rgb_white_yellow(img)
        white_yellow_images.append(result)
    park.show_images(white_yellow_images)

    # 2. 灰度化
    gray_images = []
    for img in white_yellow_images:
        gray = park.convert_gray_scale(img)
        gray_images.append(gray)
    park.show_images(gray_images)

    # 3. 边缘检测
    edge_images = []
    for img in gray_images:
        edges = park.detect_edges(img)
        edge_images.append(edges)
    park.show_images(edge_images)

    # 4. ROI选择
    roi_images = []
    for img in edge_images:
        roi = park.select_region(img)
        roi_images.append(roi)
    park.show_images(roi_images)

    # 5. 霍夫变换检测直线
    all_lines = []
    for img in roi_images:
        lines = park.hough_lines(img)
        all_lines.append(lines)

    # 6. 绘制直线
    line_images = []
    for img, lines in zip(test_images, all_lines):
        if lines is not None:
            line_img = park.draw_lines(img, lines)
            line_images.append(line_img)
    park.show_images(line_images)

    # 7. 识别停车位区域
    rect_images = []
    all_rects = []
    for img, lines in zip(test_images, all_lines):
        if lines is not None:
            rect_img, rects = park.identify_blocks(img, lines)
            rect_images.append(rect_img)
            all_rects.append(rects)
    park.show_images(rect_images)

    # 8. 绘制具体停车位
    delineated_images = []
    spot_positions = []
    for img, rects in zip(test_images, all_rects):
        delineated_img, spot_dict = park.draw_parking(img, rects)
        delineated_images.append(delineated_img)
        spot_positions.append(spot_dict)
    park.show_images(delineated_images)

    # 9. 保存结果
    final_spot_dict = spot_positions[1]  # 使用第二张图像的结果
    print(f"检测到 {len(final_spot_dict)} 个停车位")

    with open('spot_dict.pickle', 'wb') as f:
        pickle.dump(final_spot_dict, f)

    # 10. 保存CNN训练数据
    park.save_images_for_cnn(test_images[0], final_spot_dict)

    return final_spot_dict


def keras_model(weights_path):
    """加载Keras模型"""
    return load_model(weights_path, compile=False)


def img_test(test_images, final_spot_dict, model, class_dictionary):
    """测试图像"""
    for i in range(min(2, len(test_images))):  # 只测试前两张
        result = park.predict_on_image(test_images[i], final_spot_dict, model, class_dictionary)
        plt.imshow(result)
        plt.show()


def video_test(video_name, final_spot_dict, model, class_dictionary):
    """测试视频"""
    park.predict_on_video(video_name, final_spot_dict, model, class_dictionary, ret=True)


# 主程序
if __name__ == '__main__':
    # 1. 初始化
    park = Parking()

    # 2. 读取测试图像
    test_images = []
    for path in glob.glob('./test_images/*.jpg'):
        img = plt.imread(path)
        test_images.append(img)

    # 3. 显示原始图像
    park.show_images(test_images)

    # 4. 处理图像获取车位位置
    final_spot_dict = img_process(test_images, park)

    # 5. 加载模型
    model = keras_model('car1.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 6. 定义类别
    class_dictionary = {0: 'empty', 1: 'occupied'}

    # 7. 测试图像
    img_test(test_images, final_spot_dict, model, class_dictionary)

    # 8. 测试视频
    video_test('parking_video.mp4', final_spot_dict, model, class_dictionary)