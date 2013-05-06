import Image
import pic
import math
innode = []
D = []
def sigmoid(x):
	return 1/(1+math.exp(-x)) 
if __name__=='__main__':
	for i in range(10):
		p = "img/"+str(i)+".bmp"
		im = Image.open(p)
		out = pic.convertImg(im,150)
		out = pic.regulateImg(out)
		data= pic.readImg(out,32)
		f = "img/data/"+str(i)+".txt"
		handler = open(f,"w")
		for j in data:
			handler.write(str(j))
			handler.write("\n")
		handler.close()

	for i in range(10):
		p = "img/data/"+"o"+str(i)+".txt"
		f = open(p,"w")
		for j in range(10):
			if j ==i:
				f.write(str(sigmoid(i))+"\n")
			else:
				f.write(str(0)+"\n")
		f.close()

	im = Image.open("r.bmp")
	out = pic.convertImg(im,150)
	out = pic.regulateImg(out)
	data= pic.readImg(out,32)
	handler = open("r.txt","w")
	for j in data:
		handler.write(str(j))
		handler.write("\n")
	handler.close()

