@echo off
title Veri Seti Hazirlik Araci (Crop Disease)
color 0A
echo ==============================================================
echo PlantSeg Veri Seti Indiriliyor ve YOLO Formatina Ceviriliyor...
echo Dataset boyutu buyuk oldugundan bu islem internet hiziniza bagli olarak 30-40 dakika surebilir.
echo Lutfen islem bitene kadar bu pencereyi KAPATMAYIN!
echo ==============================================================
echo.

call venv\Scripts\activate

echo [1/2] Veri Seti Indiriliyor (Zenodo)...
python download_dataset.py

echo.
echo [2/2] Indirilen Veriler YOLO Formatina Filtrelenerek Ceviriliyor...
python convert_dataset.py

echo.
echo ==============================================================
echo HER SEY BASARIYLA TAMAMLANDI! 
echo Artik "python train.py" komutunu calistirarak egitimi baslatabilirsiniz.
echo ==============================================================
pause
