# 下载ZIP文件
zip_url="https://datasets.simula.no/downloads/kvasir-sessile.zip"
zip_file="kvasir-sessile.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
