#!/usr/bin/env python3
import os
import sys
import gc
import numpy as np
from osgeo import gdal
from multiprocessing import Pool

# Настройки GDAL для ускорения
gdal.SetConfigOption('GDAL_CACHEMAX', '8192')     # 8 ГБ под кэш GDAL
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

def _read_statistics_or_scan(band, band_name=""):
    """Пытается считать статистику из метаданных GDAL; если нет, 
    то делает быструю выборку по пикселям."""
    try:
        # Пробуем получить готовую статистику
        band_min, band_max, _, _ = band.GetStatistics(True, True)
        if band_min is not None and band_max is not None:
            print(f"[{band_name}] Использование GDAL-статистики: {band_min}..{band_max}")
            return band_min, band_max
    except:
        pass
    # Если не удалось - вернём None, чтобы вызвать сканирование с выборкой
    return None, None

def _scan_band_min_max(band, width, height, band_name=""):
    """Сканирует случайную выборку пикселей, чтобы вычислить min/max."""
    print(f"[{band_name}] Нет статистики GDAL, выполняем сканирование образца...")
    sample_rate = 0.01  # 1% пикселей
    sample_size = int(width * height * sample_rate)
    np.random.seed(42)  # Для воспроизводимости
    x_samples = np.random.randint(0, width, sample_size)
    y_samples = np.random.randint(0, height, sample_size)

    values = []
    for i in range(sample_size):
        x, y = x_samples[i], y_samples[i]
        pixel_val = band.ReadAsArray(x, y, 1, 1)[0, 0]
        values.append(pixel_val)

    values = np.array(values, dtype=np.float32)
    vmin, vmax = values.min(), values.max()
    print(f"[{band_name}] min={vmin}, max={vmax}")
    return vmin, vmax

# ------------------------------------------------------------------------------
# Параллельная обработка для ИНФРАКРАСНОГО КОМПОЗИТА
# ------------------------------------------------------------------------------

def _process_block_infrared(args):
    """Чтение, вычисление блочного псевдо-инфракрасного композита и возврат в виде (x, y, r, g, b, a)."""
    (x, y, real_width, real_height,
     input_file,
     nir_min, nir_max,
     red_min, red_max,
     green_min, green_max,
     nir_idx, red_idx, green_idx) = args

    # Открываем каждый раз данные в своём процессе
    ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    nir_band = ds.GetRasterBand(nir_idx)
    red_band = ds.GetRasterBand(red_idx)
    green_band = ds.GetRasterBand(green_idx)

    nir_block = nir_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
    red_block = red_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
    green_block = green_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)

    ds = None  # Закрываем файл

    # Маска фона
    background_mask = (nir_block == 0) & (red_block == 0) & (green_block == 0)

    # Подготовим выходные массивы
    r_block = np.zeros_like(nir_block, dtype=np.uint8)
    g_block = np.zeros_like(nir_block, dtype=np.uint8)
    b_block = np.zeros_like(nir_block, dtype=np.uint8)
    a_block = np.full_like(nir_block, 255, dtype=np.uint8)

    # Нормализуем
    if nir_max > nir_min:
        r_block = np.clip(((nir_block - nir_min) / (nir_max - nir_min) * 255), 0, 255).astype(np.uint8)
    if red_max > red_min:
        g_block = np.clip(((red_block - red_min) / (red_max - red_min) * 255), 0, 255).astype(np.uint8)
    if green_max > green_min:
        b_block = np.clip(((green_block - green_min) / (green_max - green_min) * 255), 0, 255).astype(np.uint8)

    # Прозрачность для фона
    a_block[background_mask] = 0

    # Освобождаем память
    del nir_block, red_block, green_block, background_mask
    gc.collect()

    return x, y, r_block, g_block, b_block, a_block

def process_infrared_by_blocks(input_file, output_file, block_size=4096, num_workers=4):
    """Создаёт псевдо-инфракрасный композит с блочной параллельной обработкой."""
    try:
        src_ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        band_count = src_ds.RasterCount

        print(f"Обрабатываем изображение (infrared) {width}x{height}, {band_count} канал(-а/-ов)")
        print(f"block_size={block_size}, потоков={num_workers}")

        # Создаём выходной файл с нужными опциями
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_file,
            width, height, 4, gdal.GDT_Byte,
            options=["BIGTIFF=YES", "COMPRESS=LZW", "PREDICTOR=2", "TILED=YES",
                     "BLOCKXSIZE=512", "BLOCKYSIZE=512"]
        )
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())

        # Устанавливаем цветовую интерпретацию
        dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        dst_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

        # Определяем индексы необходимых каналов
        # Предположим, что NIR – это 4-й канал, если доступно >=4
        nir_band_idx = 4 if band_count >= 4 else 1
        red_band_idx = 1
        green_band_idx = 2 if band_count >= 2 else 1

        # Получаем объекты полос, чтобы считать статистику
        nir_band = src_ds.GetRasterBand(nir_band_idx)
        red_band = src_ds.GetRasterBand(red_band_idx)
        green_band = src_ds.GetRasterBand(green_band_idx)

        # Пытаемся получить или вычислить min/max для каждого канала
        nir_min, nir_max = _read_statistics_or_scan(nir_band, "NIR")
        red_min, red_max = _read_statistics_or_scan(red_band, "RED")
        green_min, green_max = _read_statistics_or_scan(green_band, "GREEN")

        if nir_min is None or nir_max is None:
            nir_min, nir_max = _scan_band_min_max(nir_band, width, height, "NIR")
        if red_min is None or red_max is None:
            red_min, red_max = _scan_band_min_max(red_band, width, height, "RED")
        if green_min is None or green_max is None:
            green_min, green_max = _scan_band_min_max(green_band, width, height, "GREEN")

        src_ds = None  # Закрываем входной файл, чтобы не держать дескриптор

        # Формируем задания для пула процессов
        tasks = []
        for y in range(0, height, block_size):
            real_height = min(block_size, height - y)
            for x in range(0, width, block_size):
                real_width = min(block_size, width - x)
                tasks.append((
                    x, y, real_width, real_height,
                    input_file,
                    nir_min, nir_max,
                    red_min, red_max,
                    green_min, green_max,
                    nir_band_idx, red_band_idx, green_band_idx
                ))

        # Параллельная обработка
        print(f"Запуск параллельной обработки блоков: всего {len(tasks)} блока(ов)")
        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_block_infrared, tasks)

        # Запись результатов в выходной Dataset
        for (x, y, r_block, g_block, b_block, a_block) in results:
            dst_ds.GetRasterBand(1).WriteArray(r_block, x, y)
            dst_ds.GetRasterBand(2).WriteArray(g_block, x, y)
            dst_ds.GetRasterBand(3).WriteArray(b_block, x, y)
            dst_ds.GetRasterBand(4).WriteArray(a_block, x, y)

            del r_block, g_block, b_block, a_block
            gc.collect()

        dst_ds = None
        print(f"Успешно создан инфракрасный композит: {output_file}")
        return True

    except Exception as e:
        print(f"Ошибка в process_infrared_by_blocks: {str(e)}")
        return False

# ------------------------------------------------------------------------------
# Параллельная обработка для NDVI
# ------------------------------------------------------------------------------

def _process_block_ndvi(args):
    """Чтение, вычисление NDVI и возврат (x, y, r, g, b, a)."""
    (x, y, real_width, real_height,
     input_file,
     nir_idx, red_idx) = args

    ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    nir_band = ds.GetRasterBand(nir_idx)
    red_band = ds.GetRasterBand(red_idx)

    nir_block = nir_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
    red_block = red_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)

    ds = None

    background_mask = (nir_block == 0) & (red_block == 0)

    denominator = nir_block + red_block + 1e-6
    ndvi = (nir_block - red_block) / denominator
    normalized = (ndvi + 1.0) / 2.0  # из диапазона -1..1 в 0..1

    # RGB
    r_block = np.clip((1 - normalized) * 255, 0, 255).astype(np.uint8)
    g_block = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    b_block = np.zeros_like(r_block, dtype=np.uint8)
    a_block = np.full_like(r_block, 255, dtype=np.uint8)

    # Прозрачность фона
    a_block[background_mask] = 0

    del nir_block, red_block, ndvi, normalized, background_mask
    gc.collect()

    return x, y, r_block, g_block, b_block, a_block

def process_ndvi_by_blocks(input_file, output_file, block_size=4096, num_workers=4):
    """Создаёт NDVI с блочной параллельной обработкой."""
    try:
        src_ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        band_count = src_ds.RasterCount

        print(f"Обрабатываем изображение (NDVI) {width}x{height}, {band_count} канал(-а/-ов)")
        print(f"block_size={block_size}, потоков={num_workers}")

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_file,
            width, height, 4, gdal.GDT_Byte,
            options=["BIGTIFF=YES", "COMPRESS=LZW", "PREDICTOR=2", "TILED=YES",
                     "BLOCKXSIZE=512", "BLOCKYSIZE=512"]
        )
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())

        # Цветовая интерпретация
        dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        dst_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

        # Определяем, какие полосы использовать
        # Предположим, что 4-й канал — NIR, 1-й — RED
        nir_band_idx = 4 if band_count >= 4 else 1
        red_band_idx = 1

        src_ds = None  # Закрыли входной файл

        # Формируем задания
        tasks = []
        for y in range(0, height, block_size):
            real_height = min(block_size, height - y)
            for x in range(0, width, block_size):
                real_width = min(block_size, width - x)
                tasks.append((x, y, real_width, real_height, input_file,
                              nir_band_idx, red_band_idx))

        print(f"Запуск параллельной обработки блоков NDVI: всего {len(tasks)} блока(ов)")
        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_block_ndvi, tasks)

        # Записываем результаты
        for (x, y, r_block, g_block, b_block, a_block) in results:
            dst_ds.GetRasterBand(1).WriteArray(r_block, x, y)
            dst_ds.GetRasterBand(2).WriteArray(g_block, x, y)
            dst_ds.GetRasterBand(3).WriteArray(b_block, x, y)
            dst_ds.GetRasterBand(4).WriteArray(a_block, x, y)

            del r_block, g_block, b_block, a_block
            gc.collect()

        dst_ds = None
        print(f"Успешно создано NDVI изображение: {output_file}")
        return True

    except Exception as e:
        print(f"Ошибка в process_ndvi_by_blocks: {str(e)}")
        return False

# ------------------------------------------------------------------------------
# Параллельная обработка для VARI
# ------------------------------------------------------------------------------

def _process_block_vari(args):
    """Чтение, вычисление VARI и возврат (x, y, r, g, b, a)."""
    (x, y, real_width, real_height,
     input_file,
     red_idx, green_idx, blue_idx) = args

    ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    red_band = ds.GetRasterBand(red_idx)
    green_band = ds.GetRasterBand(green_idx)
    blue_band = ds.GetRasterBand(blue_idx)

    red_block = red_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
    green_block = green_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)
    blue_block = blue_band.ReadAsArray(x, y, real_width, real_height).astype(np.float32)

    ds = None

    background_mask = (red_block == 0) & (green_block == 0) & (blue_block == 0)

    denominator = green_block + red_block - blue_block + 1e-6
    zero_mask = np.abs(denominator) < 1e-5  # чтобы не делить на ноль

    vari = np.zeros_like(red_block, dtype=np.float32)
    valid_mask = ~zero_mask
    vari[valid_mask] = (green_block[valid_mask] - red_block[valid_mask]) / denominator[valid_mask]

    # нормализуем -1..1 в 0..1
    normalized = (vari + 1.0) / 2.0

    r_block = np.clip((1 - normalized) * 255, 0, 255).astype(np.uint8)
    g_block = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    b_block = np.zeros_like(r_block, dtype=np.uint8)
    a_block = np.full_like(r_block, 255, dtype=np.uint8)

    # Прозрачность
    a_block[background_mask | zero_mask] = 0

    del red_block, green_block, blue_block, vari, normalized
    gc.collect()

    return x, y, r_block, g_block, b_block, a_block

def process_vari_by_blocks(input_file, output_file, block_size=4096, num_workers=4):
    """Создаёт VARI изображение с блочной параллельной обработкой."""
    try:
        src_ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        band_count = src_ds.RasterCount

        print(f"Обрабатываем изображение (VARI) {width}x{height}, {band_count} канал(-а/-ов)")
        print(f"block_size={block_size}, потоков={num_workers}")

        if band_count < 3:
            print("Недостаточно каналов для вычисления VARI (нужно как минимум 3 канала: RGB).")
            return False

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_file,
            width, height, 4, gdal.GDT_Byte,
            options=["BIGTIFF=YES", "COMPRESS=LZW", "PREDICTOR=2", "TILED=YES",
                     "BLOCKXSIZE=512", "BLOCKYSIZE=512"]
        )
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())

        # Устанавливаем цветовую интерпретацию
        dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        dst_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

        # Берём первые три канала
        red_idx = 1
        green_idx = 2
        blue_idx = 3

        src_ds = None

        # Формируем задания
        tasks = []
        for y in range(0, height, block_size):
            real_height = min(block_size, height - y)
            for x in range(0, width, block_size):
                real_width = min(block_size, width - x)
                tasks.append((x, y, real_width, real_height,
                              input_file,
                              red_idx, green_idx, blue_idx))

        print(f"Запуск параллельной обработки блоков VARI: всего {len(tasks)} блока(ов)")
        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_block_vari, tasks)

        # Записываем результаты
        for (x, y, r_block, g_block, b_block, a_block) in results:
            dst_ds.GetRasterBand(1).WriteArray(r_block, x, y)
            dst_ds.GetRasterBand(2).WriteArray(g_block, x, y)
            dst_ds.GetRasterBand(3).WriteArray(b_block, x, y)
            dst_ds.GetRasterBand(4).WriteArray(a_block, x, y)

            del r_block, g_block, b_block, a_block
            gc.collect()

        dst_ds = None
        print(f"Успешно создано VARI изображение: {output_file}")
        return True

    except Exception as e:
        print(f"Ошибка в process_vari_by_blocks: {str(e)}")
        return False

# ------------------------------------------------------------------------------
# Основная точка входа
# ------------------------------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print("Использование:")
        print("  python vegetation_indices.py <метод> <входной_файл.tif> <выходной_файл.tif> [размер_блока] [число_процессов]")
        print("Методы: infrared | ndvi | vari")
        print("  размер_блока (опционально, по умолчанию 4096)")
        print("  число_процессов (опционально, по умолчанию 4)")
        sys.exit(1)

    method = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    block_size = 4096
    if len(sys.argv) >= 5:
        block_size = int(sys.argv[4])

    num_workers = 4
    if len(sys.argv) >= 6:
        num_workers = int(sys.argv[5])

    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден!")
        sys.exit(1)

    if method == "infrared":
        success = process_infrared_by_blocks(input_file, output_file, block_size, num_workers)
    elif method == "ndvi":
        success = process_ndvi_by_blocks(input_file, output_file, block_size, num_workers)
    elif method == "vari":
        success = process_vari_by_blocks(input_file, output_file, block_size, num_workers)
    else:
        print(f"Неизвестный метод: {method}")
        sys.exit(1)

    if not success:
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
