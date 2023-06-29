# 下载ZIP文件
zip_url="https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/52279/5822112/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1688266553&Signature=MnaEgrn3GjnWKxCs7uhrumderF1KNi1%2FuiVMx3OVsCqSCT5HGvdDDNADqH6eViJYuIUtaMEvAkwLgbwvunGfjpJpIXN1tVp%2BIBD8bxRa35ds7bBsdYCe1Nt1AW0viqPpq17tvYNnRRhUheMkLJACi1wiXVW9Dev6f6%2FPd5ofZnPy0vKxBIX7ZqV8y6vYrtnB0Qm8pVaRPbex2%2BeFibw%2FYnQLfhWo78DVS0iQhYgXgXoOFMFNmFyi65WPsfsBT%2FVYRM9qRF7m%2Fj9RzgCaom2zfAZPtBh45rpdyfgJBpCOUdSh2UKf7jQMRWT18uYRgHRxWQzZ8ckr%2FFMps9an6Yo3Sg%3D%3D&response-content-disposition=attachment%3B+filename%3Dhubmap-hacking-the-human-vasculature.zip"
zip_file="HuBMAP.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
