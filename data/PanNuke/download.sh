# # 下载ZIP文件
# zip_url="https://storage.googleapis.com/kaggle-data-sets/622666/3599382/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230728%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230728T070610Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4929d022a7fb6554b28fbb9407141aa5e7b925befdc7962f85f8b5d5c511d6a71837600654062cf1c4277a2627d545cb1d87c3f6466ee5d5787e84930c6b5f7df38ce6d1c2c31706b357201e9702ad4d70ac33fa24f282591ec80c0a45dbded8923f46d33a3a38b2ec5b964b03f47b9572aa19752c6b2afe901214e1eab9817fab91aac98274bc113a79d4e17876ace23d3d532119d0be88f395154024ac84d2be649fdac22854e7d643b27f4f46cc943527e43047d189e9b460723d1d150d5c599417506b68d988de10de7c9ce4d19fd3134e9c9c102b478a175b6ce188046d651731b0127eddffab0b95f55b5556fdaac7621036f050b9218eac9521c0026b"
# zip_file="Part1.zip"
# download_folder="."
# 下载ZIP文件
zip_url="https://storage.googleapis.com/kaggle-data-sets/622726/1110827/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230728%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230728T070617Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5f4f9de54239508be1867a0ea09134de72c27856c0930536a882d717c246e41c555ba9cd761f2619abcbf6a5fbbab1dada272bdae7532929c15820ad8ef6a3ced05323631a2e447e5affba81e6bada4bc6bb9e3193309ea5c8d9149fdec5baeb3ecfeb2a79036b74a8370aac4151d41d02b0c6256c5038242abf9336c644d590643a07f0c8a9a3d1c6441ed2daf4a2ae303f72d42bf40de665e906b34d909b5bdb79299935871d6b010b28e09d249d05f4c5ad6c7f35cc68e95376431a5ba82bdb716bb07c6837a5112d46a57a5d7943dba841d53bb2238b29ab97a0dbf84f03c43a9d7cd47993e9460b5a6931a21c1127a469c0dc407aa7f6eaf064a05b5bcf"
zip_file="Part2.zip"
download_folder="."
# # 下载ZIP文件
# zip_url="https://storage.googleapis.com/kaggle-data-sets/622733/1110840/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230728%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230728T070620Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4fcd72f14bbdf34cf8162c38c9ac73e7ef7046e13cd8262a7600ac774ff842f5c72ca746c94f1ff5558e047851b6ca3880cde4ae138ea3a9b2c62c50b696eedbfe3e099702eb813460551bb09c903c8ec71bbcf02e7c53d05d96d51981560484573abd8c60a94e577ed4c8bb77332eb672b9cc0d8251246ffa97a534d2a7a995981e2fb9750d3b57fab4d5853a4dc1ab782d8c27a6bcc4962f4ac6ca15e5abeab45552cccc0a5167d62efcaecf688f77a284a5a0d9112914edb9786f80598e3569f757be380e1e21be575730f6a3cadc3d5c4dd9de6979bc371c903e136c7abd4465aff67e23334aa5c02f43153d5fce2c999b8a4d070c803a7c7914fe7aeec4"
# zip_file="Part3.zip"
# download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
