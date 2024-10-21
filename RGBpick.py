import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def get_image_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

def select_points(image_path):
    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    points = []

    def onclick(event):
        if isinstance(event, matplotlib.backend_bases.KeyEvent):
            if event.key == ' ' and len(points) == 6:
                fig.canvas.mpl_disconnect(cid_click)
                fig.canvas.mpl_disconnect(cid_key)
                plt.close(fig)
            elif event.key == 'backspace' and points:
                points.pop()
                ax.clear()
                ax.imshow(img)
                for point in points:
                    ax.plot(point[0], point[1], 'w+')
        elif isinstance(event, matplotlib.backend_bases.MouseEvent) and event.button == 1:
            if event.xdata is not None and len(points) < 6:
                points.append((event.xdata, event.ydata))
                ax.plot(event.xdata, event.ydata, 'w+')
            plt.draw()

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', onclick)
    plt.show()
    # while fig.user_interaction == False:  # 等待用户完成交互
    #     plt.pause(0.1)
    # plt.close(fig)
    return points

def get_average_rgb(image, points):
    pixels = image.load()
    values = [pixels[int(x), int(y)] for x, y in points]
    average_rgb = np.mean(values, axis=0).astype(int)
    return average_rgb

def process_images():
    
    src_folder = input('请输入文件夹路径:')
    
    # 获取源文件夹中的所有文件
    files = os.listdir(src_folder)
    
    data=['filename,Red,Green,Blue\n']
    
    for file in files:
        if file.endswith('.png' or '.jpg'):
            image_path = os.path.join(src_folder, file)
            points = select_points(image_path)
            if len(points) == 6:
                image = Image.open(image_path)
                average_rgb = get_average_rgb(image, points)
                data.append(f"{file[:-4]},{average_rgb[0]},{average_rgb[1]},{average_rgb[2]}\n")
    if len(data) > 1:
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
                    file.writelines(data)  # 创建一个空文件
                    file.close()
                print(f'saved as {filename} in {src_folder}')
                break
            i += 1
    else:
        print('no image file found')
# 运行这个函数，并且提供当前目录或其他目录
process_images()
os.system('pause')