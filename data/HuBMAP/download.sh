# 下载ZIP文件
zip_url="https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/52279/5822112/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1688721277&Signature=DEvALPycfm4ArhLZqpV7ncH9rQIk4bIRcj%2Bmi5NsPQl7FT%2FE2wGVxCUGRaH39yP5WAz0wi5XyYXblJHkPN78qiC%2BKDwBfEkUMjGTRMOdyG5w%2Bhg9s3PbTY1GS6LjSGVcjGuFP7E4YQgxgi0RxV2fgdJulTRr1mI3fHBfW6nv9Dh%2Bu7BzzMpFoVcY7aeKZloLvs4e058nDS2iCYl0pAYFPGQjtpjJhMKCpiJ3W9xcBpDwfCwXTSw6gHxRKOSoCda1qCUksYN1dYwQKn%2BZG66w3k9YCi%2Fzep37N1kNE2%2BDGEHq8LN60rSOEAMv37P4UxEw2zoppPNvLUQdHfIw9cvQmA%3D%3D&response-content-disposition=attachment%3B+filename%3Dhubmap-hacking-the-human-vasculature.zip"
zip_file="HuBMAP.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
