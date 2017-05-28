import mido
from mido.parser import Parser
midi = list(mido.parser.Parser(open('test.mid')))
print len(midi)

