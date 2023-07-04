# 下载ZIP文件
zip_url="https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/18647/1126921/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1687967364&Signature=RyK4cz9K4sT8K%2BI8SooTcC3ED9YPhFUUqNnGWNAKqJkPKw75nc2WDkzi%2FDsykfp25YrJz8aAzhVugr0XSzZmF5X5kcu%2FmtO8oEyQ%2BXdZmDbd3hmsExXYPZs5oS6dzet%2BoGxRa1kl9E%2BfQE5LktTwzbLKViQWSbjTqro7NGNUuW19vc3LiMF34wO6uugROE5UUz7BwpUOiJxKiHXHl6szRMzlDN%2B65HZ5TjNf7mmJJVC2qCx1zJfNzb1WSylxwkYIf7NRR9b7O410%2FqzovNg%2B2w650F7rwm719aXru60qZ2pnW3ZbiRuDB40ZqdC%2Bazezwrk6A19h9XYa8VsYQwmVcg%3D%3D&response-content-disposition=attachment%3B+filename%3Dprostate-cancer-grade-assessment.zip"
zip_file="PanNuke.zip"
download_folder="."

wget -O "$download_folder/$zip_file" "$zip_url"
