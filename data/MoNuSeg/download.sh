# filename='MoNuSeg_test_Dataset.zip'
# fileid='1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw'
filename='MoNuSeg_train_Dataset.zip'
fileid='1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA'

wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
