import midi

pattern = midi.read_midifile("test.mid")
print (pattern[16])
print 'caca'
thefile = open('test.txt', 'w')

for line in pattern: 
  thefile.write("%s\n" % line)
