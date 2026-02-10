import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
import cv2  # 导入OpenCV库用于图像处理
import os, glob  # 导入os和glob库用于文件操作和路径匹配
import numpy as np  # 导入numpy库用于数值计算


class Parking:

    def show_images(self, images, cmap=None):
        """
        显示多张图像
        :param images: 图像列表
        :param cmap: 颜色映射，默认为None
        """
        cols = 2  # 每行显示的图像数量
        rows = (len(images) + 1) // cols  # 计算行数

        plt.figure(figsize=(15, 12))  # 设置图像显示窗口大小
        for i, image in enumerate(images):  # 遍历图像列表
            plt.subplot(rows, cols, i + 1)  # 创建子图
            cmap = 'gray' if len(image.shape) == 2 else cmap  # 如果是灰度图则使用灰度颜色映射
            plt.imshow(image, cmap=cmap)  # 显示图像
            plt.xticks([])  # 隐藏x轴刻度
            plt.yticks([])  # 隐藏y轴刻度
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)  # 调整子图间距
        plt.show()  # 显示图像

    def cv_show(self, name, img):
        """
        使用OpenCV显示图像
        :param name: 窗口名称
        :param img: 要显示的图像
        """
        cv2.imshow(name, img)  # 显示图像
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()  # 关闭所有窗口

    def select_rgb_white_yellow(self, image):
        """
        过滤掉背景，保留白色和黄色区域
        :param image: 输入图像
        :return: 过滤后的图像
        """
        lower = np.uint8([120, 120, 120])  # 定义下界颜色值
        upper = np.uint8([255, 255, 255])  # 定义上界颜色值
        white_mask = cv2.inRange(image, lower, upper)  # 创建掩码，保留指定范围内的像素
        self.cv_show('white_mask', white_mask)  # 显示掩码

        masked = cv2.bitwise_and(image, image, mask=white_mask)  # 应用掩码
        self.cv_show('masked', masked)  # 显示过滤后的图像
        return masked

    def convert_gray_scale(self, image):
        """
        将图像转换为灰度图
        :param image: 输入图像
        :return: 灰度图像
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 转换颜色空间

    def detect_edges(self, image, low_threshold=50, high_threshold=200):
        """
        检测图像边缘
        :param image: 输入图像
        :param low_threshold: Canny边缘检测低阈值
        :param high_threshold: Canny边缘检测高阈值
        :return: 边缘检测结果
        """
        return cv2.Canny(image, low_threshold, high_threshold)  # 使用Canny算法检测边缘

    def filter_region(self, image, vertices):
        """
        剔除掉不需要的区域
        :param image: 输入图像
        :param vertices: 多边形顶点坐标
        :return: 过滤后的图像
        """
        mask = np.zeros_like(image)  # 创建与图像相同大小的掩码
        if len(mask.shape) == 2:  # 如果是单通道图像
            cv2.fillPoly(mask, vertices, 255)  # 填充多边形区域
            self.cv_show('mask', mask)
        return cv2.bitwise_and(image, mask)  # 应用掩码

    def select_region(self, image):
        """
        手动选择感兴趣区域
        :param image: 输入图像
        :return: 选择区域后的图像
        """
        rows, cols = image.shape[:2]  # 获取图像尺寸
        pt_1 = [cols * 0.05, rows * 0.90]  # 定义多边形顶点
        pt_2 = [cols * 0.05, rows * 0.70]
        pt_3 = [cols * 0.30, rows * 0.55]
        pt_4 = [cols * 0.6, rows * 0.15]
        pt_5 = [cols * 0.90, rows * 0.15]
        pt_6 = [cols * 0.90, rows * 0.90]

        vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)  # 构造顶点数组
        point_img = image.copy()
        point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)  # 转换为彩色图像
        for point in vertices[0]:  # 绘制顶点
            cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
        self.cv_show('point_img', point_img)

        return self.filter_region(image, vertices)  # 返回过滤后的图像

    def hough_lines(self, image):
        """
        使用霍夫变换检测直线
        :param image: 输入图像（需为边缘检测结果）
        :return: 检测到的直线
        """
        return cv2.HoughLinesP(image, rho=0.1, theta=np.pi / 10, threshold=15, minLineLength=9,
                               maxLineGap=4)  # 霍夫变换参数设置

    def draw_lines(self, image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
        """
        绘制检测到的直线
        :param image: 输入图像
        :param lines: 直线列表
        :param color: 直线颜色
        :param thickness: 直线粗细
        :param make_copy: 是否复制图像
        :return: 绘制直线后的图像
        """
        if make_copy:
            image = np.copy(image)
        cleaned = []  # 存储符合条件的直线
        for line in lines:  # 遍历所有直线
            for x1, y1, x2, y2 in line:
                if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:  # 筛选条件
                    cleaned.append((x1, y1, x2, y2))
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)  # 绘制直线
        print("No lines detected: ", len(cleaned))  # 输出检测到的直线数量
        return image

    def identify_blocks(self, image, lines, make_copy=True):
        """
        识别停车位区域
        :param image: 输入图像
        :param lines: 直线列表
        :param make_copy: 是否复制图像
        :return: 标记停车位区域的图像和矩形坐标字典
        """
        if make_copy:
            new_image = np.copy(image)
        cleaned = []  # 存储符合条件的直线
        for line in lines:  # 筛选直线
            for x1, y1, x2, y2 in line:
                if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))

        import operator
        list1 = sorted(cleaned, key=operator.itemgetter(0, 1))  # 按照x1坐标排序

        clusters = {}  # 存储聚类结果
        dIndex = 0
        clus_dist = 10  # 聚类距离阈值

        for i in range(len(list1) - 1):  # 聚类处理
            distance = abs(list1[i + 1][0] - list1[i][0])
            if distance <= clus_dist:
                if not dIndex in clusters.keys():
                    clusters[dIndex] = []
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i + 1])
            else:
                dIndex += 1

        rects = {}  # 存储矩形区域
        i = 0
        for key in clusters:
            all_list = clusters[key]
            cleaned = list(set(all_list))  # 去重
            if len(cleaned) > 5:
                cleaned = sorted(cleaned, key=lambda tup: tup[1])  # 按y1排序
                avg_y1 = cleaned[0][1]
                avg_y2 = cleaned[-1][1]
                avg_x1 = 0
                avg_x2 = 0
                for tup in cleaned:
                    avg_x1 += tup[0]
                    avg_x2 += tup[2]
                avg_x1 = avg_x1 / len(cleaned)
                avg_x2 = avg_x2 / len(cleaned)
                rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)  # 计算平均坐标
                i += 1

        print("Num Parking Lanes: ", len(rects))  # 输出停车位车道数量
        buff = 7
        for key in rects:  # 绘制矩形框
            tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
            tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
            cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 3)
        return new_image, rects

    def draw_parking(self, image, rects, make_copy=True, color=[255, 0, 0], thickness=2, save=True):
        """
        绘制具体停车位
        :param image: 输入图像
        :param rects: 矩形区域字典
        :param make_copy: 是否复制图像
        :param color: 线条颜色
        :param thickness: 线条粗细
        :param save: 是否保存图像
        :return: 标记停车位的图像和车位字典
        """
        if make_copy:
            new_image = np.copy(image)
        gap = 15.5  # 车位间隔
        spot_dict = {}  # 车位字典
        tot_spots = 0  # 总车位数
        adj_y1 = {0: 20, 1: -10, 2: 0, 3: -11, 4: 28, 5: 5, 6: -15, 7: -15, 8: -10, 9: -30, 10: 9, 11: -32}  # y1坐标微调
        adj_y2 = {0: 30, 1: 50, 2: 15, 3: 10, 4: -15, 5: 15, 6: 15, 7: -20, 8: 15, 9: 15, 10: 0, 11: 30}  # y2坐标微调
        adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10,
                  11: 0}  # x1坐标微调
        adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}  # x2坐标微调

        for key in rects:
            tup = rects[key]
            x1 = int(tup[0] + adj_x1[key])  # 调整x1坐标
            x2 = int(tup[2] + adj_x2[key])  # 调整x2坐标
            y1 = int(tup[1] + adj_y1[key])  # 调整y1坐标
            y2 = int(tup[3] + adj_y2[key])  # 调整y2坐标
            cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制车位外框
            num_splits = int(abs(y2 - y1) // gap)  # 计算分割数量
            for i in range(0, num_splits + 1):  # 绘制水平线
                y = int(y1 + i * gap)
                cv2.line(new_image, (x1, y), (x2, y), color, thickness)
            if key > 0 and key < len(rects) - 1:  # 绘制垂直线
                x = int((x1 + x2) / 2)
                cv2.line(new_image, (x, y1), (x, y2), color, thickness)

            if key == 0 or key == (len(rects) - 1):  # 计算车位数量
                tot_spots += num_splits + 1
            else:
                tot_spots += 2 * (num_splits + 1)

            if key == 0 or key == (len(rects) - 1):  # 构建车位字典
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    spot_dict[(x1, y, x2, y + gap)] = cur_len + 1
            else:
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    x = int((x1 + x2) / 2)
                    spot_dict[(x1, y, x, y + gap)] = cur_len + 1
                    spot_dict[(x, y, x2, y + gap)] = cur_len + 2

        print("total parking spaces: ", tot_spots, cur_len)  # 输出总车位数
        if save:
            filename = 'with_parking.jpg'
            cv2.imwrite(filename, new_image)  # 保存图像
        return new_image, spot_dict

    def assign_spots_map(self, image, spot_dict, make_copy=True, color=[255, 0, 0], thickness=2):
        """
        在图像上标记车位
        :param image: 输入图像
        :param spot_dict: 车位字典
        :param make_copy: 是否复制图像
        :param color: 矩形颜色
        :param thickness: 矩形线条粗细
        :return: 标记车位后的图像
        """
        if make_copy:
            new_image = np.copy(image)
        for spot in spot_dict.keys():  # 遍历车位字典
            (x1, y1, x2, y2) = spot
            cv2.rectangle(new_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)  # 绘制矩形
        return new_image

    def save_images_for_cnn(self, image, spot_dict, folder_name='cnn_data'):
        """
        保存车位图像用于CNN训练
        :param image: 输入图像
        :param spot_dict: 车位字典
        :param folder_name: 保存文件夹名称
        """
        for spot in spot_dict.keys():  # 遍历车位字典
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            spot_img = image[y1:y2, x1:x2]  # 裁剪车位图像
            spot_img = cv2.resize(spot_img, (0, 0), fx=2.0, fy=2.0)  # 放大图像
            spot_id = spot_dict[spot]  # 获取车位ID

            filename = 'spot' + str(spot_id) + '.jpg'  # 构造文件名
            print(spot_img.shape, filename, (x1, x2, y1, y2))  # 打印调试信息

            cv2.imwrite(os.path.join(folder_name, filename), spot_img)  # 保存图像

    def make_prediction(self, image, model, class_dictionary):
        """
        使用模型进行预测
        :param image: 输入图像
        :param model: 训练好的模型
        :param class_dictionary: 类别字典
        :return: 预测标签
        """
        img = image / 255.  # 归一化图像
        image = np.expand_dims(img, axis=0)  # 转换为4D张量
        class_predicted = model.predict(image)  # 模型预测
        inID = np.argmax(class_predicted[0])  # 获取最大概率索引
        label = class_dictionary[inID]  # 获取对应标签
        return label

    def predict_on_image(self, image, spot_dict, model, class_dictionary, make_copy=True, color=[0, 255, 0], alpha=0.5):
        """
        在图像上进行车位状态预测
        :param image: 输入图像
        :param spot_dict: 车位字典
        :param model: 训练好的模型
        :param class_dictionary: 类别字典
        :param make_copy: 是否复制图像
        :param color: 空车位标记颜色
        :param alpha: 透明度
        :return: 标记车位状态后的图像
        """
        if make_copy:
            new_image = np.copy(image)
            overlay = np.copy(image)
        self.cv_show('new_image', new_image)
        cnt_empty = 0  # 空车位计数
        all_spots = 0  # 总车位计数
        for spot in spot_dict.keys():  # 遍历车位
            all_spots += 1
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            spot_img = image[y1:y2, x1:x2]  # 裁剪车位图像
            spot_img = cv2.resize(spot_img, (48, 48))  # 调整图像大小

            label = self.make_prediction(spot_img, model, class_dictionary)  # 预测车位状态
            if label == 'empty':  # 如果是空车位
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)  # 标记空车位
                cnt_empty += 1

        cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)  # 叠加标记

        cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)  # 显示可用车位数

        cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)  # 显示总车位数
        save = False

        if save:
            filename = 'with_marking.jpg'
            cv2.imwrite(filename, new_image)  # 保存图像
        self.cv_show('new_image', new_image)

        return new_image

    def predict_on_video(self, video_name, final_spot_dict, model, class_dictionary, ret=True):
        """
        在视频上进行车位状态预测
        :param video_name: 视频文件名
        :param final_spot_dict: 车位字典
        :param model: 训练好的模型
        :param class_dictionary: 类别字典
        :param ret: 是否继续循环
        """
        cap = cv2.VideoCapture(video_name)  # 打开视频文件
        count = 0
        while ret:
            ret, image = cap.read()  # 读取视频帧
            count += 1
            if count == 5:  # 每5帧处理一次
                count = 0

                new_image = np.copy(image)
                overlay = np.copy(image)
                cnt_empty = 0  # 空车位计数
                all_spots = 0  # 总车位计数
                color = [0, 255, 0]
                alpha = 0.5
                for spot in final_spot_dict.keys():  # 遍历车位
                    all_spots += 1
                    (x1, y1, x2, y2) = spot
                    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                    spot_img = image[y1:y2, x1:x2]  # 裁剪车位图像
                    spot_img = cv2.resize(spot_img, (48, 48))  # 调整图像大小

                    label = self.make_prediction(spot_img, model, class_dictionary)  # 预测车位状态
                    if label == 'empty':  # 如果是空车位
                        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)  # 标记空车位
                        cnt_empty += 1

                cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)  # 叠加标记

                cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)  # 显示可用车位数

                cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)  # 显示总车位数
                cv2.imshow('frame', new_image)  # 显示处理后的帧
                if cv2.waitKey(10) & 0xFF == ord('q'):  # 按q键退出
                    break

        cv2.destroyAllWindows()  # 关闭所有窗口
        cap.release()  # 释放视频资源
