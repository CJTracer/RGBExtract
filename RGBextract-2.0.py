import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def load_image(image_path):
    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("指定的图像路径不存在，请检查路径。")
    return img

def resize_image(img):
    # 计算新的尺寸（3/5原尺寸）
    center = (img.shape[1] // 2, img.shape[0] // 2)  # 图像中心(x, y)
    new_width = int(img.shape[1] * 3 / 5)
    new_height = int(img.shape[0] * 3 / 5)
    
    # 计算新的裁剪边界
    x_start = center[0] - new_width // 2
    x_end = center[0] + new_width // 2
    y_start = center[1] - new_height // 2
    y_end = center[1] + new_height // 2
    
    # 裁剪图像
    cropped_img = img[y_start:y_end, x_start:x_end]
    return cropped_img

def convert_to_hsv(rgb_image):
    # 将RGB图像转换为HSV
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

def threshold_hsv(hsv_image):
    # 根据HSV阈值筛选颜色，假设点滴板颜色在特定范围内
    # 这些阈值需要根据点滴板的颜色进行调整
    lower_bound = np.array([115, 0, 170])  # 例如，高亮度的低阈值
    upper_bound = np.array([179, 20, 210])  # 高亮度的高阈值
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    #检查显示掩码质量———————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    cv2.imshow('Cropped Image', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return mask

def find_contours(binary_image):
    # 寻找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours_by_area(contours, min_area=500):
    # 过滤小于指定面积的轮廓
    filtered = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    return filtered

def get_bounding_rectangles(contours):
    # 计算轮廓的边界矩形
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    return rectangles

def crop_image(image, rectangle):
    # 根据边界矩形裁剪图像
    x, y, width, height = rectangle
    return image[y:y+height, x:x+width]

def cluster_hsv(hsv_img,n):
    # 将图像数据展开成一维数组，以便进行聚类
    data = hsv_img.reshape((-1, 3))
    
    # 进行KMeans聚类
    kmeans = KMeans(n_clusters=n, n_init=4, algorithm='elkan')
    kmeans.fit(data)
    
    # 获取聚类标签
    labels = kmeans.labels_
    
    # 从KMeans对象获取聚类中心点（即每个类的HSV平均值）
    cluster_centers = kmeans.cluster_centers_
    
    # 使用聚类中心值重新构建图像
    clustered_data = cluster_centers[labels].astype('uint8')
    
    # 将聚类后的数据恢复成原始图像的形状
    clustered_image = clustered_data.reshape(hsv_img.shape)
    
    #检查聚类质量——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    cv2.imshow('clustered_image', clustered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    return clustered_image

def detect_circles_and_average_color(image, original_img, param2):
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    RGB = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(RGB, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 70, param1=0.1, param2=param2, minRadius=20, maxRadius=40)
    output = img.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        centers = [(x, y) for x, y, _ in circles]
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
    #检查圆检测质量—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    
    '''
    cv2.imshow('Cropped Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    
    rgb_values = []
    
    if circles is not None:
        if len(centers)==6:    #检测圆形数目是否为6，若否，则可能是出现干扰
            # 对于每个圆心，提取RGB值
            for center in centers:
                x, y = center
                # 提取(x, y)位置的BGR值
                rgb = img[y, x]

                # 将RGB值添加到列表中
                rgb_values.append(rgb)
            
            rgb_values=np.array(rgb_values)
            average_rgb = np.mean(rgb_values, axis=0).astype(int)
            
            '''
            print('rgb值\n',rgb_values)
            print('rgb平均值\n',average_rgb)
            '''
            
            return tuple(average_rgb.astype(int))
        else:
            return tuple('圆形数目出错，请手动提取。')
    else:
        return tuple('圆形数目出错，请手动提取。')
    
def get_cropped_img(image_path):
    img = load_image(image_path)
    img = resize_image(img)
    hsv_img = convert_to_hsv(img)
    thresh_img = threshold_hsv(hsv_img)
    contours = find_contours(thresh_img)
    filtered_contours = filter_contours_by_area(contours)
    rectangles = get_bounding_rectangles(filtered_contours)
    
    if rectangles:
        # 取最大的矩形（假设它代表点滴板）
        largest_rectangle = max(rectangles, key=lambda r: r[2] * r[3])
        cropped_img = crop_image(img, largest_rectangle)
        
        #显示二次裁剪后的图像————————————————————————————————————————————————————————————————————————————————————————————————————————————
        '''
        cv2.imshow('Cropped Image', cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return cropped_img
        
    
    
def main(image_path):
    cropped_img = get_cropped_img(image_path)
    if cropped_img is not None and cropped_img.size > 0:
        n=2
        param2=10
        clustered_image = cluster_hsv(cropped_img, n)
        average_color = detect_circles_and_average_color(clustered_image, cropped_img, param2)
        if average_color == tuple('圆形数目出错，请手动提取。'):
            n_=3
            param2_=10
            clustered_image = cluster_hsv(cropped_img, n_)
            average_color = detect_circles_and_average_color(clustered_image, cropped_img, param2_)
        if average_color == tuple('圆形数目出错，请手动提取。'):
            n__=3
            param2__=13
            clustered_image = cluster_hsv(cropped_img, n__)
            average_color = detect_circles_and_average_color(clustered_image, cropped_img, param2__)
        if average_color == tuple('圆形数目出错，请手动提取。'):
            n___=4
            param2___=13
            clustered_image = cluster_hsv(cropped_img, n___)
            average_color = detect_circles_and_average_color(clustered_image, cropped_img, param2___)
        return average_color
    else:
        return tuple('识别点滴板区域失败，请手动提取。')

# 调用主函数
def processpng():

    src_folder = input('请输入文件夹路径:')

    # 获取源文件夹中的所有文件
    files = os.listdir(src_folder)
    
    #创建data列表准备写入txt文件
    csvdata=[]
    
    # 遍历所有文件
    for file in files:
        # 只处理.png文件
        if file.endswith('.png' or '.jpg'):
            # 构造源文件的完整路径
            src_file_path = os.path.join(src_folder, file)
            rgb=main(src_file_path)
            average_rgb_for_csv=','.join(map(str, rgb))
            csvdata.append(file[:-4]+','+average_rgb_for_csv+'\n')
            print(file+'\tdone')
    i = 0
    while True:
        if i == 0:
            filename = 'result.csv'
        else:
            filename = f'result{i}.csv'

        filepath = os.path.join(src_folder, filename)

        # 检查文件是否存在
        if not os.path.exists(filepath):
            # 文件不存在，创建文件
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write('filename,Red,Green,Blue\n')
                file.writelines(csvdata)
                file.close()
            print(f'saved as {filename} in {src_folder}')
            break
        i += 1
        
processpng()
os.system('pause')