import cv2  # 导入OpenCV库，用于图像处理与计算机视觉操作


def sort_contours(cnts, method="left-to-right"):
    """
    对轮廓列表按指定方向排序（如从左到右、从上到下等）

    :param cnts: 轮廓列表，每个元素为 OpenCV 的 contour（numpy array）
    :param method: 排序方式，支持：
                   "left-to-right"（默认）、"right-to-left"、
                   "top-to-bottom"、"bottom-to-top"
    :return: (sorted_cnts, sorted_bboxes) 元组，分别表示排序后的轮廓和对应边界框
    """
    reverse = False  # 是否反向排序（如右→左、下→上需 reverse=True）
    i = 0  # 排序依据的坐标轴索引：0=x（水平），1=y（垂直）

    # 若为从右到左或从下到上，则启用反向排序
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # 若为垂直方向排序（上→下 或 下→上），则按 y 坐标（boundingBox[1]）排序
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 为每个轮廓计算最小外接矩形：返回 (x, y, w, h)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # 将轮廓与对应边界框配对，按 boundingBox[i]（x 或 y）排序
    # key=lambda b: b[1][i] → 取每个元组中 boundingBox 的第 i 个值（x 或 y）
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    等比例缩放图像

    :param image: 输入图像（numpy array）
    :param width: 目标宽度（若为 None，则按高度缩放）
    :param height: 目标高度（若为 None，则按宽度缩放）
    :param inter: 插值方法，默认为 cv2.INTER_AREA（适合缩小）
    :return: 缩放后的图像
    """
    dim = None
    (h, w) = image.shape[:2]  # 获取原图高、宽

    # 若宽高均未指定，直接返回原图
    if width is None and height is None:
        return image

    # 若仅指定高度，则按高度等比缩放宽度
    if width is None:
        r = height / float(h)  # 缩放比例
        dim = (int(w * r), height)  # 新尺寸：(新宽, 新高)
    else:
        r = width / float(w)  # 缩放比例
        dim = (width, int(h * r))  # 新尺寸：(新宽, 新高)

    # 执行缩放
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
