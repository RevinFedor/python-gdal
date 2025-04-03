#!/usr/bin/env python3
import os
import sys
import numpy as np
from osgeo import gdal

def process_infrared(input_file, output_file):
    # Явно настраиваем работу с исключениями
    gdal.UseExceptions()
    
    try:
        # Открываем входной файл
        ds = gdal.Open(input_file)
        
        # Получаем информацию о изображении
        width = ds.RasterXSize
        height = ds.RasterYSize
        band_count = ds.RasterCount
        
        print(f"Обрабатываем изображение {width}x{height} с {band_count} каналами")
        
        # Читаем необходимые каналы
        if band_count >= 4:  # Если есть NIR канал
            print("Используем 4-й канал как NIR")
            nir = ds.GetRasterBand(4).ReadAsArray().astype(np.float32)
        else:
            print("Используем 1-й канал как NIR")
            nir = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        
        print("Чтение красного канала")
        red = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        
        if band_count >= 2:  # Если есть зеленый канал
            print("Чтение зеленого канала")
            green = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
        else:
            print("Используем 1-й канал вместо зеленого")
            green = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        
        # Создаем маску для фоновых пикселей (где все каналы == 0)
        print("Создание маски фона")
        background_mask = (nir == 0) & (red == 0) & (green == 0)
        
        # Нормализуем значения векторизовано
        print("Нормализация NIR канала")
        min_nir, max_nir = np.min(nir), np.max(nir)
        r_band = np.zeros_like(nir, dtype=np.uint8)
        if max_nir > min_nir:
            r_band = np.clip(((nir - min_nir) / (max_nir - min_nir) * 255), 0, 255).astype(np.uint8)
        
        print("Нормализация красного канала")
        min_red, max_red = np.min(red), np.max(red)
        g_band = np.zeros_like(red, dtype=np.uint8)
        if max_red > min_red:
            g_band = np.clip(((red - min_red) / (max_red - min_red) * 255), 0, 255).astype(np.uint8)
        
        print("Нормализация зеленого канала")
        min_grn, max_grn = np.min(green), np.max(green)
        b_band = np.zeros_like(green, dtype=np.uint8)
        if max_grn > min_grn:
            b_band = np.clip(((green - min_grn) / (max_grn - min_grn) * 255), 0, 255).astype(np.uint8)
        
        # Создаем альфа-канал (255 для видимых пикселей, 0 для фона)
        print("Создание альфа-канала")
        alpha_band = np.full_like(nir, 255, dtype=np.uint8)
        alpha_band[background_mask] = 0
        
        # Создаем выходной файл
        print("Создание выходного файла")
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_file, width, height, 4, gdal.GDT_Byte, 
                              options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"])
        
        # Копируем геотрансформацию и проекцию
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        
        # Записываем каналы в выходной файл
        print("Запись каналов")
        out_ds.GetRasterBand(1).WriteArray(r_band)
        out_ds.GetRasterBand(2).WriteArray(g_band)
        out_ds.GetRasterBand(3).WriteArray(b_band)
        out_ds.GetRasterBand(4).WriteArray(alpha_band)
        
        # Устанавливаем интерпретацию цвета
        out_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        out_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        out_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        out_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
        
        # Закрываем файлы и освобождаем память
        out_ds = None
        ds = None
        
        print(f"Успешно создан инфракрасный композит: {output_file}")
        return True
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python infrared_script.py input_file.tif output_file.tif")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден")
        sys.exit(1)
    
    success = process_infrared(input_file, output_file)
    if not success:
        sys.exit(1)
    
    sys.exit(0)