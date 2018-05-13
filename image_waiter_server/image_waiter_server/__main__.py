from image_waiter_server import app

if __name__ == '__main__':
    app.app.run(host='0.0.0.0', port=8080)