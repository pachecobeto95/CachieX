from pymongo import MongoClient

class MongoDBManager(object):
    """Classe de Controle de conexões com o MongoDB.
    """
    def __init__(self, app):
        self.host = 'localhost:27017'
        self.database = 'MONGO_FEATURE'
        #self.user = app.config[config_prefix+'_USERNAME']
        #self.passwd = app.config[config_prefix+'_PASSWORD']
 
    def getConnection(self):
        try:
            connection = MongoClient(self.host, authSource=self.database)
            return connection
        except Exception as e:
            print("error: " + str(e))
            return None

    def getDatabase(self, connection):
        try:
            db = connection.get_database(self.database)
            return db
        except Exception as e:
            print("error: " + str(e))
            return None

    def getCollection(self, connection, collection):
        try:
            collection = self.getDatabase(connection).get_collection(collection)
            return collection
        except Exception as e:
            print("error: " + str(e))
            return None
    
    def closeConnection(self, connection):
        try:
            connection.close()
        except Exception as e:
            print("error: " + str(e))

