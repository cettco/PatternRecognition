import Image
import pic
im = Image.open("r.bmp")
out = pic.convertImg(im,150)
out = pic.regulateImg(out)
data= pic.readImg(out,32)
handler = open("r.txt","w")
for j in data:
	handler.write(str(j))
	handler.write("\n")
handler.close()