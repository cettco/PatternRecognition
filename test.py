import pic
import Image

im = Image.open("test.jpg")
out = pic.convertImg(im,150)
out = pic.regulateImg(out)
data= pic.readImg(out,32)
print data