import json
"""import json file with parameters for the CT-graph"""

class apnn_conf:
    def __init__(self,file):
        print("---------------------------------------------------")
        print("             The CT-graph environments             ")
        print("---------------------------------------------------")
        print("Reading configuration parameters from ", file)
        with open(file, 'r') as f:
            self.conf_data = json.load(f)

    def getParameters(self):
        return self.conf_data
