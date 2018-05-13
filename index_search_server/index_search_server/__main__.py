from index_search_server import app

if __name__ == '__main__':
    app.app.run(host='0.0.0.0', port=8081)

# run single container:
# sudo docker run -p 8080:8080 -v /media/roman/Other/celebA/tmp:/usr/tmp index_search_server
# /media/roman/Other/celebA/tmp - path to folder with index.ann