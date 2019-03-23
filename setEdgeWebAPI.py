from appSetEdge import app
import config

app.debug = config.DEBUG
app.run(host='0.0.0.0', port=5003)