# filename='train.zip'
# fileid='1-ApIgOOFF-rDYg4lxGvecypApN3dWW5n'
filename='valid.zip'
fileid='1wjCMtQhOIqSaoOfozS1G5hjipZcaj-6O'

wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
