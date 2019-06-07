import re
import lib

class Tweet(object):
    def __init__(self, input, output, tid, ind, inputidx=None, outputidx=None):
        self.input =  input
        self.output = output
        self.tid = tid
        self.ind = ind
        self.inputidx = inputidx
        self.outputidx = outputidx

    def __repr__(self):
        return "{}/{}:{}->{}".format(self.ind, self.tid,self.input, self.output)

    def set_inputidx(self, inputidx):
        self.inputidx = inputidx

    def set_outputidx(self, outputidx):
        self.outputidx = outputidx

    def set_input(self, input):
        self.input = input

    def set_output(self, output):
        self.output = output

class Preprocessor:
    def __init__(self):
        self.tokens = []
        self.positions = []

    def lowercase(self):
        self.tokens = [x.lower() for x in self.tokens]
        return

    def isUrl(self, token):
        regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        match = re.match(regex, token)
        if match is None:
            return False
        else:
            return True

    def filter(self):
        filtered = []
        for pos, token in enumerate(self.tokens):
            if self.isUrl(token):
                filtered.append(lib.constants.URL)
            elif token.startswith('#'):
                filtered.append(lib.constants.HASH)
            elif token.startswith('@'):
                filtered.append(lib.constants.MENTION)
            else:
                filtered.append(token)
                self.positions.append(pos)
        self.tokens = filtered
        return

    def run(self, tokens, lowercase=False):
        self.tokens = tokens
        self.positions = []
        if(lowercase):
            self.lowercase()
        self.filter()
        return self.tokens, self.positions