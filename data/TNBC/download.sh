# 下载ZIP文件
zip_url="https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip?download=1"
zip_file="TNBC.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
