import cv2
import faiss as f
import numpy as np
import pyarrow.parquet as pq
from configs import *

def build_index(emb):
    xb = emb
    d = emb.shape[1]
    # res = faiss.StandardGpuResources()
    # flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.useFloat16 = False
    # flat_config.device = 0 #check numer of this device
    # index = faiss.IndexFlatIP(d)
    # index = faiss.GpuIndexFlatL2(res, d, flat_config)
    # index = faiss.IndexLSH(d, 4*d)
    index = f.IndexFlatL2(d)
    index.add(xb)
    return index


def search(index, id_to_name, emb, k=4):
    D, I = index.search(np.expand_dims(emb, 0), k)  # actual search
    return dict(zip([id_to_name[x] for x in I[0]], D[0]))

def search_results(index, emb, image_dataset, k=4):
    results = []
    D, I = index.search(np.expand_dims(emb, 0), k)  # actual search
    for index, item in enumerate(I[0]):
        results.append({'index': item, 'distance': D[0][index], 'element': image_dataset[item]})
    return results

def search_results_2(index, emb, image_dataset):
    results = []
    output = []
    D, I = index.search(np.expand_dims(emb, 0), 4)  # actual search
    for index, item in enumerate(I[0]):
        element = image_dataset[item]
        predicted_class = element['class']
        # print("result from search: " + predicted_class)
        # if predicted_class not in output:
        output.append(predicted_class)
        # print("added item: " + predicted_class)
        results.append({'index': item, 'distance': D[0][index], 'element': element})
        if len(output) == 4:
            break

    return results

def display_results(test_image, results):
    # cv2.namedWindow('Test Results', )
    q_img = cv2.imread(test_image['image_path'])
    q_img = cv2.resize(q_img, (160, 160))
    images = []
    distances = ['Query']
    sections = [test_image['class']]
    print("results")
    for result in results:
        element = result['element']
        print(element['image_path'])
        img = cv2.imread(element['image_path'])
        img = cv2.resize(img, (160, 160))
        images.append(img)
        sections.append(element['class'])
        distances.append(result['distance'])
    imgs = np.concatenate(images, axis=1)
    f_imgs = np.concatenate((q_img, imgs), axis=1)
    b, g, r = cv2.split(f_imgs)
    black = np.zeros((30, f_imgs.shape[1], 3), dtype=b.dtype)
    f_imgs = np.concatenate((f_imgs, black), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 25
    y1 = 210
    y2 = 250
    fontScale = 1
    color_default = (255, 255, 255)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)

    lineType = 1
    f_imgs = cv2.resize(f_imgs, (0, 0), fx=1.5, fy=1.5)
    for i, string in enumerate(distances):
        color = color_default
        if str(sections[0]) == str(sections[i]) and i > 0: color = color_green
        elif str(sections[0]) != str(sections[i]) and i > 0: color = color_red

        if string != 'Query': string = np.round(string, 2)
        f_imgs = cv2.putText(f_imgs, '[' + str(sections[i]) + ']:',
                             (x, y1),
                             font,
                             fontScale,
                             color,
                             lineType)
        f_imgs = cv2.putText(f_imgs, str(string),
                             (x, y2),
                             font,
                             fontScale,
                             color,
                             lineType)
        x += 241

    # cv2.imshow('Test Results', f_imgs)
    file_name = test_image['image_name']
    path = path_as_string(results_path) + f'/knn/{file_name}.png';
    print("result save in" + path)
    cv2.imwrite(path, f_imgs)

def display_results_seq(q_path, res_path, results, threshold):
    cv2.namedWindow('Test Results', )
    q_img = cv2.imread(str(q_path))
    q_img = cv2.resize(q_img, (160, 120))
    images = []
    distances = ['Query']
    sections = [str(q_path.stem)[0:4]]
    for image_name, distance in results.items():
        img = cv2.imread(f"{res_path}/{str(image_name)[0:4]}/{image_name}.jpeg")
        img = cv2.resize(img, (160, 120))
        images.append(img)
        sections.append(str(image_name)[0:4])
        distances.append(distance)
    imgs = np.concatenate(images, axis=1)
    f_imgs = np.concatenate((q_img, imgs), axis=1)
    b, g, r = cv2.split(f_imgs)
    black = np.zeros((30, f_imgs.shape[1], 3), dtype=b.dtype)
    f_imgs = np.concatenate((f_imgs, black), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 2
    y = 210
    fontScale = 1
    color_default = (255, 255, 255)

    lineType = 1
    f_imgs = cv2.resize(f_imgs, (0, 0), fx=1.5, fy=1.5)
    for i, string in enumerate(distances):
        color = color_default

        diff = abs(int(sections[0]) - int(sections[i])) / threshold
        if diff <= 0.5 and i > 0: color = (0, 255, 255 * 2*diff)
        elif diff > 0.5 and i > 0: color = (0, max(0, 255 * (1-diff)), 255)

        if string != 'Query': string = np.round(string, 2)
        text = str('[' + str(sections[i]) + ']: ' + str(string))
        f_imgs = cv2.putText(f_imgs, text,
                             (x, y),
                             font,
                             fontScale,
                             color,
                             lineType)
        x += 241

    cv2.imshow('Test Results', f_imgs)
    cv2.imwrite(f'film/seq/img{str(sections[0])}.png', f_imgs)
    cv2.waitKey()
