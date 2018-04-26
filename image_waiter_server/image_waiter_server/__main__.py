from image_waiter_server import app

if __name__ == '__main__':
    app.app.run(host='0.0.0.0', port=8081)

# run single container:
# sudo docker run -p 8081:8081 -v /media/roman/Other/celebA:/usr/work_dir image_waiter_server
# /media/roman/Other/celebA - path to work folder with all staff