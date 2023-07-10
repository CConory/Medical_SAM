# 下载ZIP文件
zip_url="https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/30892/2715462/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1689242501&Signature=mc0frL3GV197EkCiW43on%2BzNrLQmp2PS3aWPGTaFBOQzXyeZJ5%2BSyUiryF76KtuI%2BAewrNoWF%2BAfr8DNpYv8gKMuY1ePnIFyxfGPlhbqlPO3xPnLP6j9WqLYyP4PFS0L0sBF5H89WmpOBYltJdFZQSoVmPEEdkxoFr%2FBesWjuwZKbZuD%2FoD0TADqzOUbuvhzlcfL7%2BDc%2Bmspnhrw3OiWSrOVQ%2Fv0F0LgPAeozwreCayQB8c1411Ot4gaiq%2FYwPFgvA2ROYSvL8HMj%2Fo30LykFdAlZBCHj4603kB5vrOwGMuiPFG%2BFsifc%2F9xZhaT3CbwpEAv8GbpLH8%2Bmm6V6zj9DA%3D%3D&response-content-disposition=attachment%3B+filename%3Dbkai-igh-neopolyp.zip"
zip_file="bkai-igh-neopolyp.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
