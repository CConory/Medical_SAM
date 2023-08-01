# 下载ZIP文件
zip_url="https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/30892/2715462/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1690790347&Signature=qpT2r5xtZlvH7Ccd3jJUs0%2BTSAIhBYwkkTMIktxUxI1EoZeMTCcO4HA6dsRa2G0sID5bv%2BQlP6%2BmcB6N7od4vib%2BCBc7WS4%2FYPHuAbBN3arPLAP0IX2ELzusKW%2BXb0Xi9wZKHme1Sy6Sn23BtrB0pFTD8DpY2%2FnQZm92UPmSJCh7exD56uaukBkTRMWw2uYmg2Nqig0momqo8rMBwXidtUcjCUWez3vSvOpe9swh%2BWLDocacCnTJeVER1Kzxr5XRwcEiSw1uQsC4nMb7uD5TFl4X7bwdRST6Bdq3Wc9NnXbu3W0OwWAwaexCL50so8DQ8rmK0k4Bl5DksWeCUsGkVw%3D%3D&response-content-disposition=attachment%3B+filename%3Dbkai-igh-neopolyp.zip"
zip_file="bkai-igh-neopolyp.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
