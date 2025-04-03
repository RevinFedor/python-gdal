#!/usr/bin/env python3
import os
import sys
import numpy as np
from osgeo import gdal
import gc

def process_infrared_by_blocks(input_file, output_file, block_size=2048):
    """Создает псевдо-инфракрасный композит с блочной обработкой"""
    gdal.UseExceptions()
    
    try:
        # Открываем входной файл
        src_ds = gdal.Open(input_file)
        
        # Получаем информацию о изображении
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        band_count = src_ds.RasterCount
        
        print(f"Обрабатываем изображение {width}x{height} с {band_count} каналами с использованием блочной обработки")
        
        # Создаем выходной файл
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(output_file, width, height, 4, gdal.GDT_Byte, 
                              options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"])
        
        # Копируем геотрансформацию и проекцию
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())
        
        # Устанавливаем интерпретацию цвета
        dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        dst_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
        
        # Определяем, какие полосы читать
        if band_count >= 4:
            nir_band_idx = 4
        else:
            nir_band_idx = 1
            
        red_band_idx = 1
        green_band_idx = 2 if band_count >= 2 else 1
        
        # Получаем полосы
        nir_band = src_ds.GetRasterBand(nir_band_idx)
        red_band = src_ds.GetRasterBand(red_band_idx)
        green_band = src_ds.GetRasterBand(green_band_idx)
        
        # Сначала сканируем весь файл для нахождения min/max значений
        # (можно оптимизировать, используя выборку или статистику GDAL)
        print("Вычисление статистики для нормализации...")
        
        try:
            # Попробуем использовать статистику GDAL, если она доступна
            # Это намного быстрее, чем сканировать весь файл
            nir_min, nir_max, _, _ = nir_band.GetStatistics(0, 1)
            red_min, red_max, _, _ = red_band.GetStatistics(0, 1)
            green_min, green_max, _, _ = green_band.GetStatistics(0, 1)
            
            print(f"Использование статистики GDAL: NIR({nir_min}-{nir_max}), RED({red_min}-{red_max}), GREEN({green_min}-{green_max})")
            
        except:
            print("Статистика GDAL недоступна, сканирование файла для определения min/max значений...")
            # Если статистика недоступна, используем выборку пикселей
            sample_rate = 0.01  # 1% пикселей
            sample_size = int(width * height * sample_rate)
            
            # Случайные координаты для выборки
            np.random.seed(42)  # Для воспроизводимости
            x_samples = np.random.randint(0, width, sample_size)
            y_samples = np.random.randint(0, height, sample_size)
            
            nir_values = []
            red_values = []
            green_values = []
            
            for i in range(sample_size):
                x, y = x_samples[i], y_samples[i]
                # Читаем отдельные пиксели
                nir_values.append(nir_band.ReadAsArray(x, y, 1, 1)[0, 0])
                red_values.append(red_band.ReadAsArray(x, y, 1, 1)[0, 0])
                green_values.append(green_band.ReadAsArray(x, y, 1, 1)[0, 0])
            
            nir_min, nir_max = np.min(nir_values), np.max(nir_values)
            red_min, red_max = np.min(red_values), np.max(red_values)
            green_min, green_max = np.min(green_values), np.max(green_values)
            
            print(f"Вычисленные min/max: NIR({nir_min}-{nir_max}), RED({red_min}-{red_max}), GREEN({green_min}-{green_max})")
        
        # Обработка по блокам
        for y in range(0, height, block_size):
            real_height = min(block_size, height - y)
            
            for x in range(0, width, block_size):
                real_width = min(block_size, width - x)
                
                print(f"Обработка блока: x={x}, y={y}, width={real_width}, height={real_height}")
                
                # Читаем блоки данных
                nir_block = nir_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
                red_block = red_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
                green_block = green_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
                
                # Создаем маску для фоновых пикселей
                background_mask = (nir_block == 0) & (red_block == 0) & (green_block == 0)
                
                # Нормализуем значения для инфракрасного композита
                r_block = np.zeros_like(nir_block, dtype=np.uint8)
                g_block = np.zeros_like(red_block, dtype=np.uint8)
                b_block = np.zeros_like(green_block, dtype=np.uint8)
                alpha_block = np.full_like(nir_block, 255, dtype=np.uint8)
                
                if nir_max > nir_min:
                    r_block = np.clip(((nir_block - nir_min) / (nir_max - nir_min) * 255), 0, 255).astype(np.uint8)
                if red_max > red_min:
                    g_block = np.clip(((red_block - red_min) / (red_max - red_min) * 255), 0, 255).astype(np.uint8)
                if green_max > green_min:
                    b_block = np.clip(((green_block - green_min) / (green_max - green_min) * 255), 0, 255).astype(np.uint8)
                
                # Применяем маску фона
                alpha_block[background_mask] = 0
                
                # Записываем блоки в выходной файл
                dst_ds.GetRasterBand(1).WriteArray(r_block, x, y)
                dst_ds.GetRasterBand(2).WriteArray(g_block, x, y)
                dst_ds.GetRasterBand(3).WriteArray(b_block, x, y)
                dst_ds.GetRasterBand(4).WriteArray(alpha_block, x, y)
                
                # Освобождаем память
                del nir_block, red_block, green_block, r_block, g_block, b_block, alpha_block, background_mask
                gc.collect()
        
        # Закрываем файлы и освобождаем память
        dst_ds = None
        src_ds = None
        
        print(f"Успешно создан инфракрасный композит: {output_file}")
        return True
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return False

def process_ndvi_by_blocks(input_file, output_file, block_size=2048):
    """Создает NDVI изображение с блочной обработкой"""
    gdal.UseExceptions()
    
    try:
        # Открываем входной файл
        src_ds = gdal.Open(input_file)
        
        # Получаем информацию о изображении
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        band_count = src_ds.RasterCount
        
        print(f"Обрабатываем изображение {width}x{height} с {band_count} каналами с использованием блочной обработки")
        
        # Создаем выходной файл
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(output_file, width, height, 4, gdal.GDT_Byte, 
                              options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"])
        
        # Копируем геотрансформацию и проекцию
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())
        
        # Устанавливаем интерпретацию цвета
        dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        dst_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
        
        # Определяем, какие полосы читать
        if band_count >= 4:
            nir_band_idx = 4
        else:
            nir_band_idx = 1
            
        red_band_idx = 1
        
        # Получаем полосы
        nir_band = src_ds.GetRasterBand(nir_band_idx)
        red_band = src_ds.GetRasterBand(red_band_idx)
        
        # Обработка по блокам
        for y in range(0, height, block_size):
            real_height = min(block_size, height - y)
            
            for x in range(0, width, block_size):
                real_width = min(block_size, width - x)
                
                print(f"Обработка блока NDVI: x={x}, y={y}, width={real_width}, height={real_height}")
                
                # Читаем блоки данных
                nir_block = nir_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
                red_block = red_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
                
                # Создаем маску для фоновых пикселей
                background_mask = (nir_block == 0) & (red_block == 0)
                
                # Вычисляем NDVI
                denominator = nir_block + red_block + 1e-6
                ndvi = (nir_block - red_block) / denominator
                normalized = (ndvi + 1) / 2  # Нормализуем -1..1 к 0..1
                
                # Создаем RGB представление
                r_block = np.clip((1 - normalized) * 255, 0, 255).astype(np.uint8)
                g_block = np.clip(normalized * 255, 0, 255).astype(np.uint8)
                b_block = np.zeros((real_height, real_width), dtype=np.uint8)
                alpha_block = np.full((real_height, real_width), 255, dtype=np.uint8)
                
                # Применяем маску фона
                alpha_block[background_mask] = 0
                
                # Записываем блоки в выходной файл
                dst_ds.GetRasterBand(1).WriteArray(r_block, x, y)
                dst_ds.GetRasterBand(2).WriteArray(g_block, x, y)
                dst_ds.GetRasterBand(3).WriteArray(b_block, x, y)
                dst_ds.GetRasterBand(4).WriteArray(alpha_block, x, y)
                
                # Освобождаем память
                del nir_block, red_block, ndvi, normalized, r_block, g_block, b_block, alpha_block, background_mask
                gc.collect()
        
        # Закрываем файлы и освобождаем память
        dst_ds = None
        src_ds = None
        
        print(f"Успешно создано NDVI изображение: {output_file}")
        return True
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return False

def process_vari_by_blocks(input_file, output_file, block_size=2048):
    """Создает VARI изображение с блочной обработкой"""
    gdal.UseExceptions()
    
    try:
        # Открываем входной файл
        src_ds = gdal.Open(input_file)
        
        # Получаем информацию о изображении
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        band_count = src_ds.RasterCount
        
        print(f"Обрабатываем изображение {width}x{height} с {band_count} каналами с использованием блочной обработки")
        
        # Проверяем, достаточно ли каналов
        if band_count < 3:
            print("Недостаточно каналов для вычисления VARI (нужно как минимум 3 канала RGB)")
            return False
        
        # Создаем выходной файл
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(output_file, width, height, 4, gdal.GDT_Byte, 
                              options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"])
        
        # Копируем геотрансформацию и проекцию
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())
        
        # Устанавливаем интерпретацию цвета
        dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        dst_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
        
        # Получаем полосы
        red_band = src_ds.GetRasterBand(1)
        green_band = src_ds.GetRasterBand(2)
        blue_band = src_ds.GetRasterBand(3)
        
        # Обработка по блокам
        for y in range(0, height, block_size):
            real_height = min(block_size, height - y)
            
            for x in range(0, width, block_size):
                real_width = min(block_size, width - x)
                
                print(f"Обработка блока VARI: x={x}, y={y}, width={real_width}, height={real_height}")
                
                # Читаем блоки данных
                red_block = red_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
                green_block = green_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
                blue_block = blue_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
                
                # Создаем маску для фоновых пикселей
                background_mask = (red_block == 0) & (green_block == 0) & (blue_block == 0)
                
                # Вычисляем VARI
                denominator = green_block + red_block - blue_block + 1e-6
                zero_mask = np.abs(denominator) < 1e-5
                
                # Вычисляем VARI только для ненулевых знаменателей
                vari = np.zeros_like(red_block)
                vari[~zero_mask] = (green_block[~zero_mask] - red_block[~zero_mask]) / denominator[~zero_mask]
                
                # Нормализуем от -1..1 к 0..1
                normalized = (vari + 1) / 2
                
                # Создаем RGB представление
                r_block = np.clip((1 - normalized) * 255, 0, 255).astype(np.uint8)
                g_block = np.clip(normalized * 255, 0, 255).astype(np.uint8)
                b_block = np.zeros((real_height, real_width), dtype=np.uint8)
                alpha_block = np.full((real_height, real_width), 255, dtype=np.uint8)
                
                # Применяем маску фона и нулевых знаменателей
                alpha_block[background_mask | zero_mask] = 0
                
                # Записываем блоки в выходной файл
                dst_ds.GetRasterBand(1).WriteArray(r_block, x, y)
                dst_ds.GetRasterBand(2).WriteArray(g_block, x, y)
                dst_ds.GetRasterBand(3).WriteArray(b_block, x, y)
                dst_ds.GetRasterBand(4).WriteArray(alpha_block, x, y)
                
                # Освобождаем память
                del red_block, green_block, blue_block, vari, normalized
                del r_block, g_block, b_block, alpha_block, background_mask, zero_mask
                gc.collect()
        
        # Закрываем файлы и освобождаем память
        dst_ds = None
        src_ds = None
        
        print(f"Успешно создано VARI изображение: {output_file}")
        return True
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return False

def main():
    if len(sys.argv) < 4:
        print("Использование: python vegetation_indices.py <метод> <входной_файл.tif> <выходной_файл.tif> [размер_блока]")
        print("Доступные методы:")
        print("  infrared - псевдо-инфракрасный композит")
        print("  ndvi - нормализованный разностный вегетационный индекс")
        print("  vari - видимый атмосферно-устойчивый индекс")
        print("Дополнительные параметры:")
        print("  размер_блока - размер блока для обработки в пикселях (по умолчанию 2048)")
        sys.exit(1)
    
    method = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Опциональный размер блока
    block_size = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден")
        sys.exit(1)
    
    success = False
    
    if method == "infrared":
        success = process_infrared_by_blocks(input_file, output_file, block_size)
    elif method == "ndvi":
        success = process_ndvi_by_blocks(input_file, output_file, block_size)
    elif method == "vari":
        success = process_vari_by_blocks(input_file, output_file, block_size)
    else:
        print(f"Неизвестный метод: {method}")
        print("Доступные методы: infrared, ndvi, vari")
        sys.exit(1)
    
    if not success:
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()