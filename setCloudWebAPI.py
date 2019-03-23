from appSetCloud import app
import config

app.debug = config.DEBUG
app.run(host='192.168.0.8', port=5002)