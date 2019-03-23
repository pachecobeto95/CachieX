
import json, sys, requests



jsonData = {"oi":"vixi"} 


try:
    url = 'http://localhost:5000/api/cachemiss'
    #data = json.load(jsonData)
    headers = {'Content-Type' : 'application/json'}
    r = requests.post(url, data=json.dumps(jsonData), headers=headers)
    if r.status_code != 201 and r.status_code != 200:
        raise Exception('Received an unsuccessful status code of %s' % r.status_code)
except Exception as err:
	print("Rotina Falhou: Não foi possível inserir o mapa estático.")
	print(err.args)
	sys.exit()
else:
	print("Rotina passou: Mapa inserido com sucesso.")