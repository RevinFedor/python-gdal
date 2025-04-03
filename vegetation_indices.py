#!/usr/bin/env python3
import os
import sys
import gc
import numpy as np
from osgeo import gdal
from multiprocessing import Pool

# GDAL настройки: выделяем 8 ГБ под кэш и используем все ядра
gdal.SetConfigOption('GDAL_CACHEMAX', '8192')
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

#########################################
# Функции для параллельной обработки блоков
#########################################

def process_block_infrared(args):
    (x, y, real_width, real_height, input_file,
     nir_band_idx, red_band_idx, green_band_idx,
     nir_min, nir_max, red_min, red_max, green_min, green_max) = args
    ds = gdal.Open(input_file)
    nir_band = ds.GetRasterBand(nir_band_idx)
    red_band = ds.GetRasterBand(red_band_idx)
    green_band = ds.GetRasterBand(green_band_idx)

    # Читаем блок данных и приводим к float32
    nir_block = nir_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
    red_block = red_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
    green_block = green_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)

    # Маска для фоновых пикселей
    background_mask = (nir_block == 0) & (red_block == 0) & (green_block == 0)

    # Нормализация для каждого канала
    if nir_max > nir_min:
        r_block = np.clip(((nir_block - nir_min) / (nir_max - nir_min) * 255), 0, 255).astype(np.uint8)
    else:
        r_block = np.zeros_like(nir_block, dtype=np.uint8)
    if red_max > red_min:
        g_block = np.clip(((red_block - red_min) / (red_max - red_min) * 255), 0, 255).astype(np.uint8)
    else:
        g_block = np.zeros_like(red_block, dtype=np.uint8)
    if green_max > green_min:
        b_block = np.clip(((green_block - green_min) / (green_max - green_min) * 255), 0, 255).astype(np.uint8)
    else:
        b_block = np.zeros_like(green_block, dtype=np.uint8)

    alpha_block = np.full_like(nir_block, 255, dtype=np.uint8)
    alpha_block[background_mask] = 0
    ds = None
    return (x, y, r_block, g_block, b_block, alpha_block)

def process_block_ndvi(args):
    (x, y, real_width, real_height, input_file,
     nir_band_idx, red_band_idx) = args
    ds = gdal.Open(input_file)
    nir_band = ds.GetRasterBand(nir_band_idx)
    red_band = ds.GetRasterBand(red_band_idx)

    nir_block = nir_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
    red_block = red_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)

    background_mask = (nir_block == 0) & (red_block == 0)
    denominator = nir_block + red_block + 1e-6
    ndvi = (nir_block - red_block) / denominator
    normalized = (ndvi + 1) / 2

    r_block = np.clip((1 - normalized) * 255, 0, 255).astype(np.uint8)
    g_block = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    b_block = np.zeros((real_height, real_width), dtype=np.uint8)
    alpha_block = np.full((real_height, real_width), 255, dtype=np.uint8)
    alpha_block[background_mask] = 0
    ds = None
    return (x, y, r_block, g_block, b_block, alpha_block)

def process_block_vari(args):
    (x, y, real_width, real_height, input_file) = args
    ds = gdal.Open(input_file)
    red_band = ds.GetRasterBand(1)
    green_band = ds.GetRasterBand(2)
    blue_band = ds.GetRasterBand(3)

    red_block = red_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
    green_block = green_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
    blue_block = blue_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)

    background_mask = (red_block == 0) & (green_block == 0) & (blue_block == 0)
    denominator = green_block + red_block - blue_block + 1e-6
    zero_mask = np.abs(denominator) < 1e-5

    vari = np.zeros_like(red_block)
    vari[~zero_mask] = (green_block[~zero_mask] - red_block[~zero_mask]) / denominator[~zero_mask]
    normalized = (vari + 1) / 2

    r_block = np.clip((1 - normalized) * 255, 0, 255).astype(np.uint8)
    g_block = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    b_block = np.zeros((real_height, real_width), dtype=np.uint8)
    alpha_block = np.full((real_height, real_width), 255, dtype=np.uint8)
    alpha_block[background_mask | zero_mask] = 0
    ds = None
    return (x, y, r_block, g_block, b_block, alpha_block)

#########################################
# Основные функции обработки
#########################################

def process_infrared_by_blocks(input_file, output_file, block_size=2048):
    """Создает псевдо-инфракрасный композит с блочной и параллельной обработкой"""
    gdal.UseExceptions()
    try:
        src_ds = gdal.Open(input_file)
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        band_count = src_ds.RasterCount

        print(f"Обрабатываем изображение {width}x{height} с {band_count} каналами (infrared)")
        
        # Создаем выходной файл с LZW сжатием и увеличенными размерами блока
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(output_file, width, height, 4, gdal.GDT_Byte,
                                 options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())
        dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        dst_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

        # Определяем номера каналов
        nir_band_idx = 4 if band_count >= 4 else 1
        red_band_idx = 1
        green_band_idx = 2 if band_count >= 2 else 1

        # Получаем статистику для нормализации (используем GDAL GetStatistics)
        print("Вычисление статистики для нормализации...")
        try:
            nir_min, nir_max, _, _ = src_ds.GetRasterBand(nir_band_idx).GetStatistics(0, 1)
            red_min, red_max, _, _ = src_ds.GetRasterBand(red_band_idx).GetStatistics(0, 1)
            green_min, green_max, _, _ = src_ds.GetRasterBand(green_band_idx).GetStatistics(0, 1)
            print(f"GDAL статистика: NIR({nir_min}-{nir_max}), RED({red_min}-{red_max}), GREEN({green_min}-{green_max})")
        except Exception:
            print("Статистика GDAL недоступна, необходимо реализовать альтернативный метод")
            src_ds = None
            return False

        src_ds = None  # Закрываем источник после вычисления статистики

        # Планирование блоков для параллельной обработки
        blocks = []
        for y in range(0, height, block_size):
            real_height = min(block_size, height - y)
            for x in range(0, width, block_size):
                real_width = min(block_size, width - x)
                print(f"Планируется блок: x={x}, y={y}, width={real_width}, height={real_height}")
                blocks.append((x, y, real_width, real_height, input_file,
                               nir_band_idx, red_band_idx, green_band_idx,
                               nir_min, nir_max, red_min, red_max, green_min, green_max))

        with Pool(processes=5) as pool:
            results = pool.map(process_block_infrared, blocks)

        # Запись результатов в выходной файл
        for res in results:
            x, y, r_block, g_block, b_block, alpha_block = res
            dst_ds.GetRasterBand(1).WriteArray(r_block, x, y)
            dst_ds.GetRasterBand(2).WriteArray(g_block, x, y)
            dst_ds.GetRasterBand(3).WriteArray(b_block, x, y)
            dst_ds.GetRasterBand(4).WriteArray(alpha_block, x, y)
            print(f"Записан блок: x={x}, y={y}")

        dst_ds = None
        print(f"Успешно создан инфракрасный композит: {output_file}")
        return True

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return False

def process_ndvi_by_blocks(input_file, output_file, block_size=2048):
    """Создает NDVI изображение с блочной и параллельной обработкой"""
    gdal.UseExceptions()
    try:
        src_ds = gdal.Open(input_file)
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        band_count = src_ds.RasterCount

        print(f"Обрабатываем изображение {width}x{height} с {band_count} каналами (NDVI)")
        
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(output_file, width, height, 4, gdal.GDT_Byte,
                                 options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())
        dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        dst_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

        # Определяем номера каналов для NDVI
        nir_band_idx = 4 if band_count >= 4 else 1
        red_band_idx = 1

        src_ds = None

        blocks = []
        for y in range(0, height, block_size):
            real_height = min(block_size, height - y)
            for x in range(0, width, block_size):
                real_width = min(block_size, width - x)
                print(f"Планируется блок NDVI: x={x}, y={y}, width={real_width}, height={real_height}")
                blocks.append((x, y, real_width, real_height, input_file, nir_band_idx, red_band_idx))

        with Pool(processes=5) as pool:
            results = pool.map(process_block_ndvi, blocks)

        for res in results:
            x, y, r_block, g_block, b_block, alpha_block = res
            dst_ds.GetRasterBand(1).WriteArray(r_block, x, y)
            dst_ds.GetRasterBand(2).WriteArray(g_block, x, y)
            dst_ds.GetRasterBand(3).WriteArray(b_block, x, y)
            dst_ds.GetRasterBand(4).WriteArray(alpha_block, x, y)
            print(f"Записан блок NDVI: x={x}, y={y}")

        dst_ds = None
        print(f"Успешно создано NDVI изображение: {output_file}")
        return True

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return False

def process_vari_by_blocks(input_file, output_file, block_size=2048):
    """Создает VARI изображение с блочной и параллельной обработкой"""
    gdal.UseExceptions()
    try:
        src_ds = gdal.Open(input_file)
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        band_count = src_ds.RasterCount

        print(f"Обрабатываем изображение {width}x{height} с {band_count} каналами (VARI)")
        if band_count < 3:
            print("Недостаточно каналов для вычисления VARI (нужно как минимум 3 канала RGB)")
            src_ds = None
            return False

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(output_file, width, height, 4, gdal.GDT_Byte,
                                 options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())
        dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        dst_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

        src_ds = None

        blocks = []
        for y in range(0, height, block_size):
            real_height = min(block_size, height - y)
            for x in range(0, width, block_size):
                real_width = min(block_size, width - x)
                print(f"Планируется блок VARI: x={x}, y={y}, width={real_width}, height={real_height}")
                blocks.append((x, y, real_width, real_height, input_file))

        with Pool(processes=5) as pool:
            results = pool.map(process_block_vari, blocks)

        for res in results:
            x, y, r_block, g_block, b_block, alpha_block = res
            dst_ds.GetRasterBand(1).WriteArray(r_block, x, y)
            dst_ds.GetRasterBand(2).WriteArray(g_block, x, y)
            dst_ds.GetRasterBand(3).WriteArray(b_block, x, y)
            dst_ds.GetRasterBand(4).WriteArray(alpha_block, x, y)
            print(f"Записан блок VARI: x={x}, y={y}")

        dst_ds = None
        print(f"Успешно создано VARI изображение: {output_file}")
        return True

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return False

#########################################
# Основная функция
#########################################

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
    block_size = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден")
        sys.exit(1)
    
    success = False
