# Astrocytes_Final_problem.src.functions.Common_function
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
def load_images_and_create_dataframe(folder, name_dataset):
    data = []
    image_names = natsorted(os.listdir(folder))  
    for image_name in image_names:
        image_path = os.path.join(folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_data = {
            'dataset': name_dataset,
            'image_name': image_name,
            'image_path': image_path,
            'image_size': image.shape}
        data.append(image_data)
    df = pd.DataFrame(data)
    return df

def create_folders_lists(main_path):
    event_folders = []
    activity_folders = []
    folder_names = []
    for folder_name in os.listdir(main_path):
        folder_names.append(folder_name)
        folder_path = os.path.join(main_path, folder_name)
        subfolders = os.listdir(folder_path)
        event_folder_path = os.path.join(folder_path, subfolders[0])
        activity_folder_path = os.path.join(folder_path, subfolders[1])

        event_folders.append(event_folder_path)
        activity_folders.append(activity_folder_path)

    return event_folders, activity_folders, folder_names

def find_coordinates_largest_white_area(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    largest_area = 0
    largest_area_contour = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_area_contour = contour
    
    pixel_coordinates = []
    for pixel in largest_area_contour:
        x, y = pixel[0]
        pixel_coordinates.append((x, y))  
    return pixel_coordinates

def adding_pixel_coordinates(df1, df2):
    all_pixel_coordinates = []
    for image_path in df1['image_path']:
        result_function_2 = find_coordinates_largest_white_area(image_path)  
        all_pixel_coordinates.append(result_function_2)
    df2['pixel_coordinates'] = all_pixel_coordinates
    return df2

def find_max_intensity_in_area(path, pixel_coordinates):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    max_intensity = 0

    for coordinates in pixel_coordinates:
        x, y = coordinates
        intensity = image[y, x]
        max_intensity = max(intensity, max_intensity)

    return max_intensity

def adding_max_intensity(df):
    max_intensities = []
    for image_path, pixel_coordinates in zip(df['image_path'], df['pixel_coordinates']):
        result_function_3 = find_max_intensity_in_area(image_path, pixel_coordinates)  
        max_intensities.append(result_function_3)
    df['max_intensity'] = max_intensities
    return df

def plot_graph_intensity(df):
    df['time_seconds'] = df.index / 2
    
    gragh = plt.figure(figsize=(6.7, 3.11), dpi=300)   
    plt.plot(df['time_seconds'], df['max_intensity'], color='#800020', linewidth=3)
    plt.xlabel('Время, c')
    plt.ylabel('Максимальная интенсивность')
    plt.title('Максимальная интенсивность в зависимости от времени')
    plt.xlim(df['time_seconds'].min(), df['time_seconds'].max())
    plt.ylim(df['max_intensity'].min(), df['max_intensity'].max())
    plt.legend(['Максимальная интенсивность'], loc='upper right')
    plt.tight_layout()
    plt.show()
    return gragh

def save_graph(gragh, name_graph_png, name_graph_svg):
    gragh.savefig(name_graph_png)
    gragh.savefig(name_graph_svg)

def save_data_max_intensity(df, name_table):
    time_max_intensity_df = df[['time_seconds', 'max_intensity']].copy()
    print(time_max_intensity_df)
    time_max_intensity_df.to_excel(name_table)

def function_runs_all_functions(main_path):
    event_folders, activity_folders, folder_names  = create_folders_lists(main_path)
    
    for i in range(len(event_folders)):
        event_path = event_folders[i]
        activity_path = activity_folders[i]
        events_df = load_images_and_create_dataframe(event_path, 'event')
        activities_df = load_images_and_create_dataframe(activity_path, 'activity')
        events_df = adding_information_about_white_areas(events_df)
        graph_average = graph_average_area(events_df)
        folder_name = folder_names[i]
        path_to_save = rf'C:\Users\Vector\Desktop\Final\Результаты работы функций\{folder_name}'
        os.chdir(path_to_save)
        save_graph(graph_average, f'График средней площади_{i + 1}.png', f'График средней площади_{i + 1}.svg')
        save_data_average_area(events_df, f'Таблица средней площади_{i + 1}.xlsx')

        activities_df = adding_pixel_coordinates(events_df, activities_df)
        activities_df = adding_max_intensity(activities_df)
        graph_intensity = plot_graph_intensity(activities_df)
        save_graph(graph_intensity, f'График максимальной интенсивности_{i + 1}.png', f'График максимальной интенсивности_{i + 1}.svg')
        save_data_max_intensity(activities_df, f'Таблица максимальной интенсивности_{i + 1}.xlsx')