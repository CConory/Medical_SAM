# 下载ZIP文件
zip_url="https://datasets.simula.no/downloads/kvasir-seg.zip"
zip_file="Kvasir-SEG.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
