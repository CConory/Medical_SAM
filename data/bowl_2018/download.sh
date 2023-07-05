# 下载ZIP文件
zip_url="https://storage.googleapis.com/kagglesdsdata/competitions/8089/44321/stage1_train.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1688814502&Signature=ioOjyAv5gskF5ztP%2FYEookhKiRNzhe1dtxW02ednPVADrQHjG5ZiYRCGT9%2FZVXS2uvlrlAjX2imdGtA5UwVV%2FwGX6%2FZGT6JFnNEFhcU%2BQU%2BpkRmt2wFrNQ5tZQLtL4%2BCGkfLJko4%2FNll9e4HG3zlgjg1xR4nlKcNpfhyrFqlnw92aAd8m8fKSFNW5bVAmqgD4hEBAV%2FAl82nZsSCQmSWi1D6Dni6cAWxO5zCzKrlYjC4cIbi7rKvWvsDPNXdW3uftJOjZF2p60sFOyDzSfjSKd0XNJ%2Fh7fwy6FJC7%2FwBKQjOXaq3TVXb03H6%2F1IwhFB8tp56oJuAn%2Bzp6u3seLA3kw%3D%3D&response-content-disposition=attachment%3B+filename%3Dstage1_train.zip"
zip_file="bowl_2018.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
