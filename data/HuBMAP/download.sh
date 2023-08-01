# 下载ZIP文件
zip_url="https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/52279/5822112/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1688721476&Signature=aZ7AYSCSI3xNcJAYRRhpt6WugZu%2BbwGYphmHbWPNRH21o%2FjyHpWt9YiMvT1%2FEiO4CRftoZLlGMIfaf3lNxLD7WLjmnoqRT7CQhYDaiXZXek0IIpXNsEt7pJ%2FyzKt60tRIIkg8dGNbZqS4%2BRP2WWTlYWVMo7CeVuEzTT5N0GeJhVLvGE4V2HUYeK8dbCXNvwkYgcZoMiGjfVLV%2FDNiiFoNW2d6cQSPfsdMnVmtJslXBeEkL70LU1xG7uGCbxhWzqAEW%2Fr6Dy7FI0InZ6HgVY7RBOmSZwJYA11BLdzfZx8amtagSMJCNVMzbGvnxPOYq8vNq332FAWzeB211YQvGOB6Q%3D%3D&response-content-disposition=attachment%3B+filename%3Dhubmap-hacking-the-human-vasculature.zip"
zip_file="HuBMAP.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
