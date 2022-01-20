import os

if __name__ == "__main__":
	"change the name of the file to compute the total training time"
	with open('./1gpu/5606207.out') as f:
		s = 0
		ms = 0
		m = 0
		for l in f.readlines():
			if 'Elapsed' in l:
				l = l.split('Elapsed training time: 0:')[1]
				m += int(l.split(':')[0])
				k = l.split(':')[1]
				s += int(k.split('.')[0])
				ms += int(l.split('.')[1][:2])
		print('minutes ' + str(m))
		print('seconds ' + str(s))
		print('ms ' + str(ms))
