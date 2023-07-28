# 下载ZIP文件
zip_url="https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/8089/44321/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1690790098&Signature=eaG9MC3ixq2JOsTlESyed3zmt%2FU6XJ1xuEe%2BR%2Bb5BGVc5%2FBwWdwWIK3oeCZBNnhR3zlxCHmif4J2anv%2FC%2Fd5UOanuXnmt8wosck60QOu%2FFqdhChmn2%2FBcjKyFPKjZeXG4yfbZFZ%2FTw9svbV9nziJ2haGTeXeaqkfq%2B1mdv3LUhcZKMJ%2BIwSjgZ90nc1MsnEoTnrX3e9UoeGVpTyyqH6ay%2FD9Sis1%2Fj7FiyAXZDAfN6rdwcY3wCQkKo4SQRcu2cH9iwdLk6Ay0i2u9mlFPqL1FiZitrahwXA904CelCQZrYhuHoZC4emdua7%2BlX237LwgoBP3mU%2FtgB12ERI6PoYc9g%3D%3D&response-content-disposition=attachment%3B+filename%3Ddata-science-bowl-2018.zip"
zip_file="bowl_2018.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
