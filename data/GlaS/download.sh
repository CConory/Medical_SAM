# 下载ZIP文件
zip_url="https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip"
zip_file="GlaS.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
